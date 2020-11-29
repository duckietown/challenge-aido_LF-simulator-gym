import gc
import math
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import cast, Dict, Iterator, List, Optional, Tuple

import cv2
import geometry
import numpy as np
import yaml
from geometry import se2_from_linear_angular, SE2value
from zuper_commons.types import ZException, ZValueError
from zuper_nodes import TimeSpec, timestamp_from_seconds, TimingInfo
from zuper_nodes_wrapper import Context

from aido_agents.utils_leds import get_blinking_LEDs_emergency
from aido_schemas import (
    DB20Commands,
    DB20Observations,
    DB20Odometry,
    DB20RobotObservations,
    DB20SetRobotCommands,
    DTSimDuckieInfo,
    DTSimDuckieState,
    DTSimRobotInfo,
    DTSimRobotState,
    DTSimState,
    DTSimStateDump,
    EpisodeStart,
    GetDuckieState,
    GetRobotObservations,
    GetRobotState,
    JPGImage,
    LEDSCommands,
    Metric,
    PerformanceMetrics,
    PWMCommands,
    RGB,
    RobotConfiguration,
    RobotInterfaceDescription,
    RobotName,
    RobotPerformance,
    SetMap,
    SimulationState,
    SpawnDuckie,
    SpawnRobot,
    Step,
    Termination,
)
from duckietown_world import (
    construct_map,
    DuckietownMap,
    DynamicModel,
    get_lane_poses,
    GetLanePoseResult,
    iterate_by_class,
    IterateByTestResult,
    MapFormat1Constants,
    PlacedObject,
    PlatformDynamicsFactory,
    Tile,
)
from duckietown_world.world_duckietown.dynamics_delay import DelayedDynamics
from duckietown_world.world_duckietown.pwm_dynamics import get_DB18_nominal
from duckietown_world.world_duckietown.tile import translation_from_O3
from duckietown_world.world_duckietown.tile_map import ij_from_tilename
from duckietown_world.world_duckietown.utils import relative_pose
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.objects import DuckiebotObj, DuckieObj
from gym_duckietown.objmesh import get_mesh
from gym_duckietown.simulator import (
    FrameBufferMemory,
    get_duckiebot_mesh,
    NotInLane,
    ROBOT_LENGTH,
    ROBOT_WIDTH,
    SAFETY_RAD_MULT,
    Simulator,
    WHEEL_DIST,
)
from . import logger

CODE_OUT_OF_LANE = "out-of-lane"
CODE_OUT_OF_TILE = "out-of-tile"
CODE_COLLISION = "collision"


@dataclass
class GymDuckiebotSimulatorConfig:
    """
        env_constructor: either "Simulator" or "DuckietownEnv"

        env_parameters: parameters for the constructor

        camera_frame_rate: frame rate for the camera. No observations
        will be generated quicker than this.

    """

    env_constructor: str = "Simulator"
    env_parameters: dict = None
    camera_dt: float = 1 / 15.0
    render_dt: float = 1 / (15.0 * 7)
    minimum_physics_dt: float = 1 / 200.0
    blur_time: float = 0.05
    topdown_resolution: int = 640

    terminate_on_ool: bool = False
    """ Terminate on out of lane """

    terminate_on_out_of_tile: bool = True
    """ Terminate on out of tile """

    terminate_on_collision: bool = True
    collision_threshold: float = 0.15
    """ Terminate on collision """

    debug_no_video: bool = False
    """ If true, it skips the rendering and gives back a black image"""

    debug_profile: bool = False
    """ Profile the rendering and other gym operations"""

    debug_profile_summary: bool = False
    """ Make a summary of the performance """


def is_on_a_tile(dw: PlacedObject, q: SE2value) -> Optional[Tuple[int, int]]:
    it: IterateByTestResult
    for it in iterate_by_class(dw, Tile):
        tile = it.object
        coords = ij_from_tilename(it.fqn[-1])
        tile_transform = it.transform_sequence
        tile_relative_pose = relative_pose(tile_transform.asmatrix2d().m, q)
        p = translation_from_O3(tile_relative_pose)
        if tile.get_footprint().contains(p):
            return coords
    return None


class R:
    obj: DuckiebotObj
    termination: Optional[Termination]

    def __init__(self, obj: DuckiebotObj):
        self.obj = obj
        self.termination = None


class NPC(R):
    pass


class Duckie:
    obj: DuckieObj
    pose: SE2value

    def __init__(self, obj: DuckieObj, pose: SE2value):
        self.obj = obj
        self.pose = pose


class PC(R):
    spawn_configuration: RobotConfiguration
    last_commands: DB20Commands
    obs: DB20Observations

    last_observations: Optional[np.ndarray]

    last_observations_time: float

    state: DelayedDynamics
    blur_time: float
    camera_dt: float

    # history of "raw" observations and their timestamps
    render_observations: List[np.array] = field(default_factory=list)
    render_timestamps: List[float] = field(default_factory=list)
    controlled_by_player: bool

    pdf: PlatformDynamicsFactory

    def __init__(
        self,
        obj: DuckiebotObj,
        spawn_configuration: RobotConfiguration,
        pdf: PlatformDynamicsFactory,
        controlled_by_player: bool,
        blur_time: float,
        camera_dt: float,
    ):

        R.__init__(self, obj=obj)
        self.camera_dt = camera_dt
        self.blur_time = blur_time
        self.controlled_by_player = controlled_by_player
        black = RGB(0.0, 1.0, 0.0)
        leds = LEDSCommands(black, black, black, black, black)
        wheels = PWMCommands(0.0, 0.0)
        self.last_commands = DB20Commands(LEDS=leds, wheels=wheels)
        self.last_observations = None

        self.last_observations_time = -1000
        self.render_observations = []
        self.render_timestamps = []
        self.spawn_configuration = spawn_configuration

        q = spawn_configuration.pose
        v = spawn_configuration.velocity
        c0 = q, v
        self.pdf = pdf
        self.state = cast(DelayedDynamics, pdf.initialize(c0=c0, t0=0))

    def integrate(self, dt: float):
        self.state = self.state.integrate(dt, self.last_commands.wheels)

    def update_observations(self, context: Context, current_time: float):
        # context.info(f'update_observations() at {current_time}')
        assert self.render_observations

        dt = current_time - self.last_observations_time
        if dt < self.camera_dt:
            return

        to_average = []
        n = len(self.render_observations)
        # context.info(str(self.render_timestamps))
        # context.info(f'now {self.current_time}')
        for i in range(n):
            ti = self.render_timestamps[i]

            if math.fabs(current_time - ti) <= self.blur_time:
                to_average.append(self.render_observations[i])

            # need to remove the old stuff, otherwise memory grows unbounded
            if math.fabs(current_time - ti) > 5:
                self.render_observations[i] = None

        if not to_average:
            msg = "Cannot find observations to average"
            raise ZException(
                msg,
                current_time=current_time,
                render_timestamps=list(reversed(self.render_timestamps))[:10],
                blur_time=self.blur_time,
            )

        if len(to_average) == 1:
            obs = to_average[0]
        else:
            obs0 = to_average[0].astype("float32")

            for obs in to_average[1:]:
                obs0 += obs
            obs = obs0 / len(to_average)

        obs = obs.astype("uint8")
        if self.termination is not None:
            obs = rgb2grayed(obs)
            font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
            cv2.putText(obs, "Wasted", (165, 100), font, 3, (255, 0, 0), 2, cv2.LINE_AA)

        # context.info(f'update {obs.shape} {obs.dtype}')
        jpg_data = rgb2jpg(obs)
        camera = JPGImage(jpg_data)
        s: DelayedDynamics = self.state
        sd = cast(DynamicModel, s.state)
        resolution_rad = sd.parameters.encoder_resolution_rad
        odometry = DB20Odometry(
            axis_right_rad=sd.axis_right_obs_rad,
            axis_left_rad=sd.axis_left_rad,
            resolution_rad=resolution_rad,
        )
        self.obs = DB20Observations(camera, odometry)
        self.last_observations_time = current_time


def rgb2grayed(rgb):
    r = rgb[:, :, 0].squeeze()
    g = rgb[:, :, 1].squeeze()
    b = rgb[:, :, 2].squeeze()
    # note we keep a uint8
    gray = r * 299.0 / 1000 + g * 587.0 / 1000 + b * 114.0 / 1000
    gray = gray.astype("uint8")

    res = np.zeros(shape=rgb.shape, dtype="uint8")
    res[:, :, 0] = gray
    res[:, :, 1] = gray
    res[:, :, 2] = gray

    return res


def get_min_render_dt(speed: float, angular_deg: float, camera_dt: float) -> float:
    fov_deg = 60.0
    pixels_fov = 640
    pixels_deg = pixels_fov / fov_deg
    max_pixel_mov = 3
    angular_pixel_mov_sec = np.abs(angular_deg) * pixels_deg

    D = 0.3
    H = 0.1
    beta0 = np.arctan(D / H)
    beta1 = np.arctan((D + speed * 1.0 / H))
    hori_motion_apparent_motion_deg_s = beta1 - beta0
    linear_pixel_mov_sec = hori_motion_apparent_motion_deg_s * pixels_deg * 2

    current_pixel_mov_sec = linear_pixel_mov_sec + angular_pixel_mov_sec

    # fps = current_pixel_mov_sec / max_pixel_mov
    # current_pixel_mov_sec   = * dt <= max_pixel_mov
    dt_max = min(max_pixel_mov / current_pixel_mov_sec, camera_dt / 2)
    return dt_max


class GymDuckiebotSimulator:
    config: GymDuckiebotSimulatorConfig = GymDuckiebotSimulatorConfig()
    current_time: float
    reward_cumulative: float
    # name of the current episode
    episode_name: str
    # current sim time
    current_time: float
    # last time we rendered the observations
    last_render_time: float

    env: Simulator

    pcs: Dict[RobotName, PC]
    npcs: Dict[RobotName, NPC]
    duckies: Dict[RobotName, Duckie]

    dm: DuckietownMap

    def __init__(self):
        self.clear()

    def clear(self):
        self.npcs = {}
        self.pcs = {}
        self.duckies = {}

    def init(self, context: Context):
        env_parameters = self.config.env_parameters or {}
        logger.info(config=self.config)
        environment_class = self.config.env_constructor
        name2class = {
            "DuckietownEnv": DuckietownEnv,
            "Simulator": Simulator,
        }
        if not environment_class in name2class:
            msg = "Could not find environment class."
            raise ZException(msg, environment_class=environment_class, available=list(name2class))

        klass = name2class[environment_class]
        msg = "creating environment"
        logger.info(msg, environment_class=environment_class, env_parameters=env_parameters)
        env = klass(**env_parameters)
        self.set_env(env)

    def set_env(self, env):
        self.env = env

    def on_received_seed(self, context: Context, data: int):
        context.info(f"Using seed = {data!r}")
        random.seed(data)
        np.random.seed(data)

    def on_received_clear(self):
        self.clear()

    def on_received_set_map(self, data: SetMap):
        yaml_str = cast(str, data.map_data)

        map_data = yaml.load(yaml_str, Loader=yaml.SafeLoader)
        self.dm = construct_map(map_data)
        # noinspection PyProtectedMember
        self.env._interpret_map(map_data)

    def on_received_spawn_duckie(self, data: SpawnDuckie):
        q = data.pose
        pos, angle = self.env.weird_from_cartesian(q)

        mesh = get_mesh("duckie")
        kind = "duckie"
        height = 0.08
        static = True

        obj_desc = {
            "kind": kind,
            "mesh": mesh,
            "pos": pos,
            "rotate": np.rad2deg(angle),
            "height": height,
            "angle": angle,
            "static": static,
            "optional": False,
            "scale": height / mesh.max_coords[1],
        }
        obj = DuckieObj(obj_desc, domain_rand=False, safety_radius_mult=SAFETY_RAD_MULT, walk_distance=0.0)

        self.duckies[data.name] = Duckie(obj, data.pose)

    def on_received_spawn_robot(self, data: SpawnRobot):
        q = data.configuration.pose
        pos, angle = self.env.weird_from_cartesian(q)
        mesh = get_duckiebot_mesh(data.color)

        height = 0.12  # XXX

        kind = MapFormat1Constants.KIND_DUCKIEBOT
        static = True  # data.motion == MOTION_PARKED

        obj_desc = {
            "kind": kind,
            "mesh": mesh,
            "pos": pos,
            "rotate": np.rad2deg(angle),
            "height": height,
            "angle": angle,
            "static": static,
            "optional": False,
            "scale": height / mesh.max_coords[1],
        }

        obj = DuckiebotObj(obj_desc, False, SAFETY_RAD_MULT, WHEEL_DIST, ROBOT_WIDTH, ROBOT_LENGTH)

        if data.playable:
            pdf: PlatformDynamicsFactory = get_DB18_nominal(delay=0.15)  # TODO: parametric
            pc = PC(
                obj=obj,
                spawn_configuration=data.configuration,
                pdf=pdf,
                blur_time=self.config.blur_time,
                camera_dt=self.config.camera_dt,
                controlled_by_player=data.owned_by_player,
            )

            self.pcs[data.robot_name] = pc
        else:

            self.npcs[data.robot_name] = NPC(obj)

    def on_received_get_robot_interface_description(self, context: Context, data: RobotName):
        rid = RobotInterfaceDescription(robot_name=data, observations=JPGImage, commands=PWMCommands)
        context.write("robot_interface_description", rid)

    def on_received_get_robot_performance(self, context: Context, data: RobotName):
        metrics = {}
        metrics["survival_time"] = Metric(
            higher_is_better=True, cumulative_value=self.current_time, description="Survival time."
        )
        # metrics["reward"] = Metric(
        #     higher_is_better=True,
        #     cumulative_value=self.reward_cumulative,
        #     description="Cumulative agent reward.",
        # )
        pm = PerformanceMetrics(metrics)
        rid = RobotPerformance(robot_name=data, t_effective=self.current_time, performance=pm)
        context.write("robot_performance", rid)

    def on_received_episode_start(self, context: Context, data: EpisodeStart):
        self.current_time = 0.0
        self.last_render_time = -100

        S = self.config.topdown_resolution
        self.td = FrameBufferMemory(width=S, height=S)

        # self.reward_cumulative = 0.0
        self.episode_name = data.episode_name

        try:
            self.env.start_pose = [[0.0, 0.0, 0.0], 0.0]
            self.env.reset()
        except BaseException as e:
            msg = "Could not initialize environment"
            raise Exception(msg) from e

        for robot, pc in self.pcs.items():
            q, v = pc.state.TSE2_from_state()
            lprs: List[GetLanePoseResult] = list(get_lane_poses(self.dm, q))
            if not lprs:
                msg = f"Robot {robot} is out of the lane."
                raise ZException(msg, robot=robot, pc=pc)

            # verify_pose_validity(context, self.env, pc.spawn_configuration)
            self.env.objects.append(pc.obj)

        for _, npc in self.npcs.items():
            self.env.objects.append(npc.obj)

        for _, duckie in self.duckies.items():
            self.env.objects.append(duckie.obj)

        self.render(context)
        for _, pc in self.pcs.items():
            pc.update_observations(context, self.current_time)

    def on_received_step(self, context: Context, data: Step):
        profile_enabled = self.config.debug_profile
        delta_time = data.until - self.current_time
        step = f"stepping forward {int(delta_time * 1000)} s of simulation time"
        t0 = time.time()
        with timeit(step, context, min_warn=0, enabled=True):
            if delta_time > 0:
                step = "update_physics_and_observations"
                with timeit(step, context, min_warn=0, enabled=profile_enabled):
                    self.update_physics_and_observations(until=data.until, context=context)
            else:
                context.warning(f"Already at time {data.until}")

            # step = "on_received_step/_compute_done_reward"
            # with timeit(step, context, min_warn=0, enabled=profile_enabled):
            #     # noinspection PyProtectedMember
            #     d = self.env._compute_done_reward()
            # self.reward_cumulative += d.reward * delta_time
            self.current_time = data.until
        dt_real = time.time() - t0

        if self.config.debug_profile_summary:
            ratio = delta_time / dt_real
            msg = f"""
                Stepping forward {delta_time:.3f} s of simulation time took {dt_real:.3f} seconds.
                *If this was the only step* (excluding, e.g. how long it takes the agent to compute)
                then the simulation speedup is {ratio:.2f} x
            """
            context.info(msg)

        gc.collect()

    def update_physics_and_observations(self, until: float, context: Context):
        # we are at self.current_time and need to update until "until"
        # sensor_dt = self.config.camera_dt
        physics_dt = self.config.minimum_physics_dt
        # XXX
        pc0 = list(self.pcs.values())[0]
        last_observations_time = pc0.last_observations_time
        snapshots = list(get_snapshots(last_observations_time, self.current_time, until, physics_dt))

        steps = snapshots + [until]
        # context.info(f'current time: {self.current_time}')
        # context.info(f'       until: {until}')
        # context.info(f'    last_obs: {self.last_observations_time}')
        # context.info(f'   snapshots: {snapshots}')
        profile_enabled = self.config.debug_profile

        for i, t1 in enumerate(steps):
            delta_time = t1 - self.current_time

            # we "update" the action in the simulator, but really
            # we are going to move the robot ourself
            last_action = np.array([0.0, 0.0])
            # step = f"update_physics_and_observations/step{i}/update_physics"
            # with timeit(step, context, min_warn=0, enabled=True):
            #     self.env.update_physics(last_action, delta_time=delta_time)

            for pc_name, pc in self.pcs.items():
                pc.integrate(delta_time)

            self.current_time = t1

            # every render_dt, render the observations
            # if self.current_time - self.last_render_time > render_dt:
            step = f"update_physics_and_observations/step{i}/render"
            with timeit(step, context, min_warn=0, enabled=profile_enabled):
                self.render(context)
                # self.last_render_time = self.current_time

            # if self.current_time - last_observations_time >= sensor_dt:
            for pc_name, pc in self.pcs.items():
                pc.update_observations(context, self.current_time)

    def set_positions_and_commands(self, protagonist: RobotName):
        self.env.cur_pos = [-100.0, -100.0, -100.0]
        for pc_name, pc in self.pcs.items():
            q, v = pc.state.TSE2_from_state()
            cur_pos, cur_angle = self.env.weird_from_cartesian(q)

            if pc_name == protagonist:
                cur_pos[0] -= 100  # Z
                cur_pos[1] -= 100  # Z

            pc.obj.pos = cur_pos
            pc.obj.angle = cur_angle
            pc.obj.y_rot = np.rad2deg(cur_angle)

            if pc.termination:
                set_gym_leds(pc.obj, get_blinking_LEDs_emergency(self.current_time))
            else:
                if pc.last_commands is not None:
                    set_gym_leds(pc.obj, pc.last_commands.LEDS)

        for npc_name, npc in self.npcs.items():
            if npc.termination:
                set_gym_leds(npc.obj, get_blinking_LEDs_emergency(self.current_time))

    def render(self, context: Context):
        profile_enabled = self.config.debug_profile

        for i, (pc_name, pc) in enumerate(self.pcs.items()):
            self.set_positions_and_commands(protagonist=pc_name)

            # set the pose of this robot as the "protagonist"
            q, v = pc.state.TSE2_from_state()
            cur_pos, cur_angle = self.env.weird_from_cartesian(q)
            self.env.cur_pos = cur_pos
            self.env.cur_angle = cur_angle
            if pc.render_timestamps:
                dt = self.current_time - pc.render_timestamps[-1]
                linear, angular = geometry.linear_angular_from_se2(v)
                angular_deg = np.rad2deg(angular)

                speed = linear[0]

                dt_max = get_min_render_dt(speed, angular_deg, pc.camera_dt)

                do_it = dt >= dt_max
                # if do_it:
                #     context.debug(
                #         f'{pc_name} t {self.current_time:.4f} dt {dt:.3f} dt_max {dt_max:.3f} ({1 / dt:.1f} fps) w {angular_deg:.1f} '
                #         f'deg/s {do_it}')
            else:
                do_it = True

            if do_it:
                if self.config.debug_no_video:
                    obs = np.zeros((480, 640, 3), "uint8")
                else:
                    step = f"render/{i}-pc_name/render_obs"
                    with timeit(step, context, min_warn=0, enabled=profile_enabled):
                        obs = self.env.render_obs()

                pc.render_observations.append(obs)
                pc.render_timestamps.append(self.current_time)

            self.set_positions_and_commands(protagonist="")

    def on_received_set_robot_commands(self, data: DB20SetRobotCommands, context: Context):
        robot_name = data.robot_name
        wheels = data.commands.wheels
        l, r = wheels.motor_left, wheels.motor_right

        if max(math.fabs(l), math.fabs(r)) > 1:
            msg = (
                f"Received invalid PWM commands. They should be between -1 and +1."
                f" Received left = {l!r}, right = {r!r}."
            )
            context.error(msg)
            raise Exception(msg)
        if self.pcs[robot_name].termination is not None:
            context.info(f"Robot {robot_name} is terminated so inputs are ignored.")
            data.commands.wheels.motor_left = 0.0
            data.commands.wheels.motor_right = 0.0
            data.commands.LEDs = get_blinking_LEDs_emergency(self.current_time)
        self.pcs[robot_name].last_commands = data.commands

    def on_received_get_robot_observations(self, context: Context, data: GetRobotObservations):
        step = f"on_received_get_robot_observations"
        with timeit(step, context, min_warn=0, enabled=self.config.debug_profile):
            robot_name = data.robot_name
            if not robot_name in self.pcs:
                msg = f"Cannot compute observations for non-pc {robot_name!r}"
                raise ZValueError(msg, robot_name=robot_name, pcs=list(self.pcs), npcs=list(self.npcs))
            pc = self.pcs[robot_name]
            ro = DB20RobotObservations(robot_name, pc.last_observations_time, pc.obs)
            # logger.info('simulator sends', ro=ro)
            # timing information
            t = timestamp_from_seconds(pc.last_observations_time)
            ts = TimeSpec(time=t, frame=self.episode_name, clock=context.get_hostname())
            timing = TimingInfo(acquired={"image": ts})
            context.write("robot_observations", ro, with_schema=True, timing=timing)

    def _get_duckie_state(self, duckie_name: str) -> DTSimDuckieState:
        d = self.duckies[duckie_name]
        state = DTSimDuckieInfo(pose=d.pose, velocity=np.zeros((3, 3)))

        return DTSimDuckieState(duckie_name=duckie_name, t_effective=self.current_time, state=state)

    def _get_robot_state(self, robot_name: RobotName) -> DTSimRobotState:
        env = self.env
        if robot_name in self.pcs:
            pc = self.pcs[robot_name]
            q, v = pc.state.TSE2_from_state()
            state = DTSimRobotInfo(
                pose=q, velocity=v, leds=pc.last_commands.LEDS, pwm=pc.last_commands.wheels,
            )
            rs = DTSimRobotState(robot_name=robot_name, t_effective=self.current_time, state=state)
        elif robot_name in self.npcs:
            npc = self.npcs[robot_name]
            # copy from simualtor
            obj: DuckiebotObj = npc.obj
            q = env.cartesian_from_weird(obj.pos, obj.angle)
            # FIXME: how to get velocity?
            v = geometry.se2_from_linear_angular([0, 0], 0)

            def get(name) -> RGB:
                c = obj.leds_color[name]
                return RGB(float(c[0]), float(c[1]), float(c[2]))

            leds = LEDSCommands(
                front_right=get("front_right"),
                front_left=get("front_left"),
                center=get("center"),
                back_left=get("back_left"),
                back_right=get("back_right"),
            )
            wheels = PWMCommands(0.0, 0.0)  # XXX
            state = DTSimRobotInfo(pose=q, velocity=v, leds=leds, pwm=wheels,)
            rs = DTSimRobotState(robot_name=robot_name, t_effective=self.current_time, state=state)
        else:
            msg = f"Cannot compute robot state for {robot_name!r}"
            raise ZValueError(msg, robot_name=robot_name, pcs=list(self.pcs), npcs=list(self.npcs))
        return rs

    def on_received_get_robot_state(self, context: Context, data: GetRobotState):
        robot_name = data.robot_name
        rs = self._get_robot_state(robot_name)
        # timing information
        t = timestamp_from_seconds(self.current_time)
        ts = TimeSpec(time=t, frame=self.episode_name, clock=context.get_hostname())
        timing = TimingInfo(acquired={"state": ts})
        context.write("robot_state", rs, timing=timing)  # , with_schema=True)

    def on_received_get_duckie_state(self, context: Context, data: GetDuckieState):
        duckie_name = data.duckie_name
        rs = self._get_duckie_state(duckie_name)
        # timing information
        t = timestamp_from_seconds(self.current_time)
        ts = TimeSpec(time=t, frame=self.episode_name, clock=context.get_hostname())
        timing = TimingInfo(acquired={"state": ts})
        context.write("duckie_state", rs, timing=timing)  # , with_schema=True)

    def on_received_dump_state(self, context: Context):
        duckiebots = {}
        for robot_name in self.pcs:
            duckiebots[robot_name] = self._get_robot_state(robot_name).state
        for robot_name in self.npcs:
            duckiebots[robot_name] = self._get_robot_state(robot_name).state
        duckies = {}
        for duckie_name in self.duckies:
            duckies[duckie_name] = self._get_duckie_state(duckie_name).state
        simstate = DTSimState(t_effective=self.current_time, duckiebots=duckiebots, duckies=duckies)
        res = DTSimStateDump(simstate)
        context.write("state_dump", res)

    def on_received_get_sim_state(self, context: Context):

        all_robots = {}
        all_robots.update(self.pcs)
        all_robots.update(self.npcs)
        robot_states: Dict[str, DTSimRobotState]
        robot_states = {k: self._get_robot_state(k) for k in all_robots}

        the_robot: R
        for robot_name, the_robot in all_robots.items():
            if the_robot.termination is not None:
                continue

            state = robot_states[robot_name]
            q = state.state.pose
            terminate_on_static_collision = True
            if terminate_on_static_collision:
                cur_pos, cur_angle = self.env.weird_from_cartesian(q)
                # noinspection PyProtectedMember
                collided = self.env._check_intersection_static_obstacles(cur_pos, cur_angle)
                # logger.info(cur_pos=cur_pos, cur_angle=cur_angle, col=self.env.collidable_corners,
                #             collided=collided)
                if collided:
                    msg = f"Robot {robot_name!r} collided with static obstacles."
                    termination = Termination(when=self.current_time, desc=msg, code=CODE_OUT_OF_LANE)
                    the_robot.termination = termination
                    halt_robot(the_robot, q)
                    logger.error(robot_name=robot_name, termination=termination)

            if self.config.terminate_on_ool:
                lprs: List[GetLanePoseResult] = list(get_lane_poses(self.dm, q))
                if not lprs:
                    msg = f"Robot {robot_name!r} is out of the lane."
                    termination = Termination(when=self.current_time, desc=msg, code=CODE_OUT_OF_LANE)
                    the_robot.termination = termination
                    halt_robot(the_robot, q)
                    logger.error(robot_name=robot_name, termination=termination)

            if self.config.terminate_on_out_of_tile:
                tile_coords = is_on_a_tile(self.dm, q)
                if tile_coords is None:
                    msg = f"Robot {robot_name!r} is out of tile."
                    termination = Termination(when=self.current_time, desc=msg, code=CODE_OUT_OF_TILE)
                    the_robot.termination = termination
                    halt_robot(the_robot, q)
                    logger.error(robot_name=robot_name, termination=termination)
            if self.config.terminate_on_collision:
                for duckie_name, duckie in self.duckies.items():
                    d = relative_pose(duckie.pose, q)
                    dist = np.linalg.norm(translation_from_O3(d))
                    if dist < self.config.collision_threshold:
                        msg = f"Robot {robot_name!r} collided with duckie {duckie_name!r}."
                        termination = Termination(when=self.current_time, desc=msg, code=CODE_COLLISION)
                        the_robot.termination = termination
                        halt_robot(the_robot, q)
                        logger.error(robot_name=robot_name, termination=termination)

                for other_robot, its_state in robot_states.items():
                    if other_robot == robot_name:
                        continue
                    q2 = its_state.state.pose
                    d = relative_pose(q2, q)
                    dist = np.linalg.norm(translation_from_O3(d))
                    if dist < self.config.collision_threshold:
                        msg = f"Robot {robot_name!r} collided with {other_robot!r}"
                        termination = Termination(when=self.current_time, desc=msg, code=CODE_COLLISION)
                        the_robot.termination = termination
                        halt_robot(the_robot, q)
                        logger.error(robot_name=robot_name, termination=termination)

        robots_owned_by_player = [k for k, v in self.pcs.items() if v.controlled_by_player]
        terminated_robots = [
            k for k, v in (list(self.pcs.items()) + list(self.npcs.items())) if v.termination is not None
        ]
        all_player_robots_terminated = len(set(robots_owned_by_player) - set(terminated_robots)) == 0
        done = all_player_robots_terminated
        terminations = {k: v.termination for k, v in all_robots.items()}
        sim_state = SimulationState(done, terminations=terminations)
        context.write("sim_state", sim_state)

    def on_received_get_ui_image(self, context: Context):
        self.set_positions_and_commands(protagonist="")
        profile_enabled = self.config.debug_profile
        S = self.config.topdown_resolution, self.config.topdown_resolution
        if self.config.debug_no_video:
            shape = S[0], S[1], 3
            top_down_observation = np.zeros(shape, "uint8")
        else:
            step = "on_received_get_ui_image/render_top_down"
            with timeit(step, context, min_warn=0, enabled=profile_enabled):
                # noinspection PyProtectedMember
                td = self.td

                # noinspection PyProtectedMember
                top_down_observation = self.env._render_img(
                    width=td.width,
                    height=td.height,
                    multi_fbo=td.multi_fbo,
                    final_fbo=td.final_fbo,
                    img_array=td.img_array,
                    top_down=True,
                )

        jpg_data = rgb2jpg(top_down_observation)
        jpg = JPGImage(jpg_data)
        context.write("ui_image", jpg)


def set_gym_leds(obj: DuckiebotObj, LEDS: LEDSCommands):
    obj.leds_color["front_right"] = get_rgb_tuple(LEDS.front_right)
    obj.leds_color["front_left"] = get_rgb_tuple(LEDS.front_left)
    obj.leds_color["back_right"] = get_rgb_tuple(LEDS.back_right)
    obj.leds_color["back_left"] = get_rgb_tuple(LEDS.back_left)
    obj.leds_color["center"] = get_rgb_tuple(LEDS.center)


def get_snapshots(last_obs_time: float, current_time: float, until: float, dt: float) -> Iterator[float]:
    t = last_obs_time + dt
    while t < until:
        if t > current_time:
            yield t
        t += dt


def rgb2jpg(rgb: np.ndarray) -> bytes:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    compress = cv2.imencode(".jpg", bgr)[1]
    jpg_data = np.array(compress).tobytes()
    return jpg_data


@contextmanager
def timeit(s, context, min_warn=0.01, enabled=True):
    t0 = time.time()
    yield

    if not enabled:
        return
    t1 = time.time()

    delta = t1 - t0
    msg = f"timeit: {int((t1 - t0) * 1000):4d} ms for {s}"
    if delta > min_warn:
        context.info(msg)


def halt_robot(r: R, pose: SE2value):
    if isinstance(r, PC):
        pdf: PlatformDynamicsFactory = get_DB18_nominal(delay=0.15)  # TODO: parametric
        v0 = se2_from_linear_angular([0.0, 0.0], 0.0)
        c0 = pose, v0
        r.state = pdf.initialize(c0=c0, t0=0)


def verify_pose_validity(context: Context, env: Simulator, spawn_configuration):
    q = spawn_configuration.pose
    cur_pos, cur_angle = env.weird_from_cartesian(q)
    q2 = env.cartesian_from_weird(cur_pos, cur_angle)

    # okaysh : set at least one robot to the pose
    env.cur_pos = cur_pos
    env.cur_angle = cur_angle

    i, j = env.get_grid_coords(env.cur_pos)
    # noinspection PyProtectedMember
    tile = env._get_tile(i, j)

    msg = ""
    msg += f"\ni, j: {i}, {j}"
    msg += f"\nPose: {geometry.SE2.friendly(q)}"
    msg += f"\nPose: {geometry.SE2.friendly(q2)}"
    msg += f"\nCur pos: {cur_pos}"
    context.info(msg)

    if tile is None:
        msg = "Current pose is not in a tile: \n" + msg
        raise Exception(msg)

    kind = tile["kind"]
    is_straight = kind.startswith("straight")

    context.info(f'Sampled tile  {tile["coords"]} {tile["kind"]} {tile["angle"]}')

    if not is_straight:
        context.info("not on a straight tile")

    # noinspection PyProtectedMember
    valid = env._valid_pose(np.array(cur_pos), cur_angle)
    context.info(f"valid: {valid}")

    try:
        lp = env.get_lane_pos2(cur_pos, cur_angle)
        context.info(f"Sampled lane pose {lp}")
        context.info(f"dist: {lp.dist}")
    except NotInLane:
        raise

    if not valid:
        msg = "Not valid"
        context.error(msg)


def get_rgb_tuple(x: RGB) -> Tuple[float, float, float]:
    return x.r, x.g, x.b
