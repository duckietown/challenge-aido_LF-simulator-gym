#!/usr/bin/env python3
#
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
from geometry import SE2value
from zuper_commons.logs import setup_logging, ZLogger
from zuper_commons.types import ZException, ZValueError
from zuper_nodes import TimeSpec, timestamp_from_seconds, TimingInfo
from zuper_nodes_wrapper import Context, wrap_direct

from aido_schemas import (DB20Commands, DB20Observations, DB20Odometry, DB20RobotObservations,
                          DB20SetRobotCommands, DTSimRobotInfo, DTSimRobotState, DTSimState, DTSimStateDump,
                          EpisodeStart, GetRobotObservations, GetRobotState, JPGImage, LEDSCommands, Metric,
                          PerformanceMetrics, protocol_simulator_DB20, PWMCommands, RGB, RobotConfiguration,
                          RobotInterfaceDescription, RobotName, RobotPerformance, SetMap, SimulationState,
                          SpawnRobot, Step)
from aido_schemas.protocol_simulator import MOTION_PARKED
from duckietown_world import (construct_map, DuckietownMap, DynamicModel, get_lane_poses, GetLanePoseResult,
                              iterate_by_class, IterateByTestResult, PlacedObject, PlatformDynamicsFactory,
                              Tile)
from duckietown_world.world_duckietown.dynamics_delay import DelayedDynamics
from duckietown_world.world_duckietown.pwm_dynamics import get_DB18_nominal
from duckietown_world.world_duckietown.tile import translation_from_O3
from duckietown_world.world_duckietown.tile_map import ij_from_tilename
from duckietown_world.world_duckietown.utils import relative_pose
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.graphics import create_frame_buffers
from gym_duckietown.objects import DuckiebotObj
from gym_duckietown.simulator import (NotInLane, ObjMesh, ROBOT_LENGTH, ROBOT_WIDTH, SAFETY_RAD_MULT,
                                      Simulator,
                                      WHEEL_DIST)

logger = ZLogger('gym_bridge')
__version__ = '6.0.11'

logger.info(f'gym_bridge {__version__}')


@dataclass
class GymDuckiebotSimulatorConfig:
    """
        env_constructor: either "Simulator" or "DuckietownEnv"

        env_parameters: parameters for the constructor

        camera_frame_rate: frame rate for the camera. No observations
        will be generated quicker than this.

    """
    env_constructor: str = 'Simulator'
    env_parameters: dict = None
    camera_dt: float = 1 / 15.0
    render_dt: float = 1 / (15.0 * 7)
    minimum_physics_dt: float = 1 / 30.0
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


def is_on_a_tile(dw: PlacedObject, q: SE2value, tol=0.000001) -> Optional[Tuple[int, int]]:
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

    def __init__(self, obj: DuckiebotObj):
        self.obj = obj


class NPC(R):
    pass


class PC(R):
    spawn_configuration: RobotConfiguration
    # ro: DB18RobotObservations = None
    last_commands: DB20Commands
    obs: DB20Observations

    last_observations: np.array

    last_observations_time: float = None

    state: DelayedDynamics
    blur_time: float

    # history of "raw" observations and their timestamps
    render_observations: List[np.array] = field(default_factory=list)
    render_timestamps: List[float] = field(default_factory=list)

    def __init__(self, obj: DuckiebotObj, spawn_configuration,
                 pdf: PlatformDynamicsFactory,
                 blur_time: float):
        R.__init__(self, obj=obj)
        self.blur_time = blur_time
        black = RGB(.0, .0, .0)
        leds = LEDSCommands(black, black, black, black, black)
        wheels = PWMCommands(.0, .0)
        self.last_commands = DB20Commands(LEDS=leds, wheels=wheels)
        self.last_observations = None

        self.last_observations_time = -1000
        self.render_observations = []
        self.render_timestamps = []
        self.spawn_configuration = spawn_configuration

        q = spawn_configuration.pose
        v = spawn_configuration.velocity
        c0 = q, v

        self.state = cast(DelayedDynamics, pdf.initialize(c0=c0, t0=0))

    def integrate(self, dt: float):
        self.state = self.state.integrate(dt, self.last_commands.wheels)

    def update_observations(self, context: Context, current_time: float):
        # context.info(f'update_observations() at {current_time}')
        assert self.render_observations

        to_average = []
        n = len(self.render_observations)
        # context.info(str(self.render_timestamps))
        # context.info(f'now {self.current_time}')
        for i in range(n):
            ti = self.render_timestamps[i]

            if math.fabs(current_time - ti) < self.blur_time:
                to_average.append(self.render_observations[i])

        try:
            obs0 = to_average[0].astype('float32')

            for obs in to_average[1:]:
                obs0 += obs
            obs = obs0 / len(to_average)
        except IndexError:
            obs = self.render_observations[0]
        obs = obs.astype('uint8')
        # context.info(f'update {obs.shape} {obs.dtype}')
        jpg_data = rgb2jpg(obs)
        camera = JPGImage(jpg_data)
        s: DelayedDynamics = self.state
        sd = cast(DynamicModel, s.state)
        resolution_rad = sd.parameters.encoder_resolution_rad
        odometry = DB20Odometry(axis_right_rad=sd.axis_right_obs_rad,
                                axis_left_rad=sd.axis_left_rad,
                                resolution_rad=resolution_rad)
        self.obs = DB20Observations(camera, odometry)
        self.last_observations_time = current_time


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

    # reward_cumulative: float

    env: Simulator

    pcs: Dict[RobotName, PC]
    npcs: Dict[RobotName, NPC]
    dm: DuckietownMap

    def __init__(self):
        self.clear()

    def clear(self):
        self.npcs = {}
        self.pcs = {}

    def init(self):
        env_parameters = self.config.env_parameters or {}
        print(f"render_dt: {self.config.render_dt}")
        environment_class = self.config.env_constructor
        name2class = {
            'DuckietownEnv': DuckietownEnv,
            'Simulator': Simulator,
        }
        if not environment_class in name2class:
            msg = 'Could not find environment class {} in {}'.format(environment_class, list(name2class))
            raise Exception(msg)

        klass = name2class[environment_class]
        logger.info('creating environment', environment_class=environment_class,
                    env_parameters=env_parameters)
        self.env = klass(**env_parameters)

    def on_received_seed(self, context: Context, data: int):
        context.info(f'Using seed = {data!r}')
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

    def on_received_spawn_robot(self, data: SpawnRobot):
        q = data.configuration.pose
        pos, angle = self.env.weird_from_cartesian(q)

        mesh = ObjMesh.get('duckiebot')
        height = 0.12
        if data.playable:
            kind = 'duckiebot-player'
            static = True
        else:
            kind = 'duckiebot'

            static = data.motion == MOTION_PARKED

        obj_desc = {
            'kind': kind,
            'mesh': mesh,
            'pos': pos,
            'rotate': np.rad2deg(angle),
            'height': height,
            'y_rot': np.rad2deg(angle),
            'static': static,
            'optional': False,
            'scale': height / mesh.max_coords[1]
        }

        obj = DuckiebotObj(obj_desc, False, SAFETY_RAD_MULT, WHEEL_DIST,
                           ROBOT_WIDTH, ROBOT_LENGTH)

        if data.playable:
            pdf: PlatformDynamicsFactory = get_DB18_nominal(delay=0.15)  # TODO: parametric
            pc = PC(obj=obj, spawn_configuration=data.configuration, pdf=pdf,
                    blur_time=self.config.blur_time)

            self.pcs[data.robot_name] = pc
        else:

            self.npcs[data.robot_name] = NPC(obj)

    def on_received_get_robot_interface_description(self, context: Context, data: RobotName):
        rid = RobotInterfaceDescription(robot_name=data, observations=JPGImage, commands=PWMCommands)
        context.write('robot_interface_description', rid)

    def on_received_get_robot_performance(self, context: Context, data: RobotName):
        metrics = {}
        metrics['survival_time'] = Metric(higher_is_better=True, cumulative_value=self.current_time,
                                          description="Survival time.")
        metrics['reward'] = Metric(higher_is_better=True, cumulative_value=self.reward_cumulative,
                                   description="Cumulative agent reward.")
        pm = PerformanceMetrics(metrics)
        rid = RobotPerformance(robot_name=data, t_effective=self.current_time, performance=pm)
        context.write('robot_performance', rid)

    def on_received_episode_start(self, context: Context, data: EpisodeStart):
        self.current_time = 0.0
        self.last_render_time = -100
        self.reward_cumulative = 0.0
        self.episode_name = data.episode_name
        self.top_down_observation = None
        TOPDOWN_SIZE = self.config.topdown_resolution, self.config.topdown_resolution
        self.top_down_multi_fbo, self.top_down_final_fbo = create_frame_buffers(
            TOPDOWN_SIZE[0],
            TOPDOWN_SIZE[1],
            4
        )

        self.top_down_img_array = np.zeros(shape=(TOPDOWN_SIZE[0], TOPDOWN_SIZE[1], 3), dtype=np.uint8)

        try:
            self.env.reset()
        except BaseException as e:
            msg = 'Could not initialize environment'
            raise Exception(msg) from e

        for robot, pc in self.pcs.items():
            q, v = pc.state.TSE2_from_state()
            lprs: List[GetLanePoseResult] = list(get_lane_poses(self.dm, q))
            if not lprs:
                msg = f'Robot {robot} is out of the lane.'
                raise ZException(msg, robot=robot, pc=pc)

            verify_pose_validity(context, self.env, pc.spawn_configuration)
            self.env.objects.append(pc.obj)

        for _, npc in self.npcs.items():
            self.env.objects.append(npc.obj)

        self.render(context)
        for _, pc in self.pcs.items():
            pc.update_observations(context, self.current_time)

    def on_received_step(self, context: Context, data: Step):
        profile_enabled = self.config.debug_profile
        delta_time = data.until - self.current_time
        step = f'stepping forward {int(delta_time * 1000)} s of simulation time'
        t0 = time.time()
        with timeit(step, context, min_warn=0, enabled=True):

            if delta_time > 0:
                step = 'update_physics_and_observations'
                with timeit(step, context, min_warn=0, enabled=profile_enabled):
                    self.update_physics_and_observations(until=data.until, context=context)
            else:
                context.warning(f'Already at time {data.until}')

            step = 'on_received_step/_compute_done_reward'
            with timeit(step, context, min_warn=0, enabled=profile_enabled):
                # noinspection PyProtectedMember
                d = self.env._compute_done_reward()
            self.reward_cumulative += d.reward * delta_time
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

    def update_physics_and_observations(self, until: float, context: Context):
        # we are at self.current_time and need to update until "until"
        sensor_dt = self.config.camera_dt
        render_dt = self.config.render_dt
        # XXX
        pc0 = list(self.pcs.values())[0]
        last_observations_time = pc0.last_observations_time
        snapshots = list(get_snapshots(last_observations_time, self.current_time, until, render_dt))

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
            step = f'update_physics_and_observations/step{i}/update_physics'
            with timeit(step, context, min_warn=0, enabled=profile_enabled):
                self.env.update_physics(last_action, delta_time=delta_time)

            for pc_name, pc in self.pcs.items():
                pc.integrate(delta_time)

            self.current_time = t1

            # every render_dt, render the observations
            if self.current_time - self.last_render_time > render_dt:
                step = f'update_physics_and_observations/step{i}/render'
                with timeit(step, context, min_warn=0, enabled=profile_enabled):
                    self.render(context)
                self.last_render_time = self.current_time

            if self.current_time - last_observations_time >= sensor_dt:
                for pc_name, pc in self.pcs.items():
                    pc.update_observations(context, self.current_time)

    def set_positions_and_commands(self, protagonist: str):
        for pc_name, pc in self.pcs.items():
            q, v = pc.state.TSE2_from_state()
            cur_pos, cur_angle = self.env.weird_from_cartesian(q)

            if pc_name == protagonist:
                cur_pos[1] -= 1  # Z

            pc.obj.pos = cur_pos
            pc.obj.angle = cur_angle
            pc.obj.y_rot = np.rad2deg(cur_angle)

            if pc.last_commands is not None:
                pc.obj.leds_color['front_right'] = get_rgb_tuple(pc.last_commands.LEDS.front_right)
                pc.obj.leds_color['front_left'] = get_rgb_tuple(pc.last_commands.LEDS.front_left)
                pc.obj.leds_color['back_right'] = get_rgb_tuple(pc.last_commands.LEDS.back_right)
                pc.obj.leds_color['back_left'] = get_rgb_tuple(pc.last_commands.LEDS.back_left)
                pc.obj.leds_color['center'] = get_rgb_tuple(pc.last_commands.LEDS.center)

    def render(self, context: Context):
        # context.info(f'render() at {self.current_time}')

        # for each robot that needs observations
        profile_enabled = self.config.debug_profile

        for i, (pc_name, pc) in enumerate(self.pcs.items()):
            self.set_positions_and_commands(protagonist=pc_name)

            # set the pose of this robot as the "protagonist"
            q, v = pc.state.TSE2_from_state()
            cur_pos, cur_angle = self.env.weird_from_cartesian(q)
            self.env.cur_pos = cur_pos
            self.env.cur_angle = cur_angle

            # render the observations

            if self.config.debug_no_video:
                obs = np.zeros((480, 640, 3), 'uint8')
            else:
                step = f'render/{i}-pc_name/render_obs'
                with timeit(step, context, min_warn=0, enabled=profile_enabled):
                    obs = self.env.render_obs()
            # context.info(f'render {obs.shape} {obs.dtype}')
            pc.render_observations.append(obs)
            pc.render_timestamps.append(self.current_time)

            self.set_positions_and_commands(protagonist="")

    def on_received_set_robot_commands(self, data: DB20SetRobotCommands, context: Context):
        robot_name = data.robot_name
        wheels = data.commands.wheels
        l, r = wheels.motor_left, wheels.motor_right

        if max(math.fabs(l), math.fabs(r)) > 1:
            msg = f'Received invalid PWM commands. They should be between -1 and +1.' \
                  f' Received left = {l!r}, right = {r!r}.'
            context.error(msg)
            raise Exception(msg)
        self.pcs[robot_name].last_commands = data.commands

    def on_received_get_robot_observations(self, context: Context, data: GetRobotObservations):
        step = f'on_received_get_robot_observations '
        with timeit(step, context, min_warn=0, enabled=self.config.debug_profile):
            robot_name = data.robot_name
            if not robot_name in self.pcs:
                msg = f'Cannot compute observations for non-pc {robot_name!r}'
                raise ZValueError(msg, robot_name=robot_name, pcs=list(self.pcs),
                                  npcs=list(self.npcs))
            pc = self.pcs[robot_name]
            ro = DB20RobotObservations(robot_name, pc.last_observations_time, pc.obs)
            # logger.info('simulator sends', ro=ro)
            # timing information
            t = timestamp_from_seconds(pc.last_observations_time)
            ts = TimeSpec(time=t, frame=self.episode_name, clock=context.get_hostname())
            timing = TimingInfo(acquired={'image': ts})
            context.write('robot_observations', ro, with_schema=True, timing=timing)

    def _get_robot_state(self, robot_name: RobotName) -> DTSimRobotState:
        env = self.env
        if robot_name in self.pcs:
            pc = self.pcs[robot_name]
            q, v = pc.state.TSE2_from_state()
            state = DTSimRobotInfo(pose=q,
                                   velocity=v,
                                   leds=pc.last_commands.LEDS,
                                   pwm=pc.last_commands.wheels,
                                   )
            rs = DTSimRobotState(robot_name=robot_name,
                                 t_effective=self.current_time,
                                 state=state)
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

            leds = LEDSCommands(front_right=get('front_right'),
                                front_left=get('front_left'),
                                center=get('center'),
                                back_left=get('back_left'),
                                back_right=get('back_right'))
            wheels = PWMCommands(0.0, 0.0)  # XXX
            state = DTSimRobotInfo(pose=q,
                                   velocity=v,
                                   leds=leds,
                                   pwm=wheels,
                                   )
            rs = DTSimRobotState(robot_name=robot_name,
                                 t_effective=self.current_time,
                                 state=state)
        else:
            msg = f'Cannot compute robot state for {robot_name!r}'
            raise ZValueError(msg, robot_name=robot_name, pcs=list(self.pcs),
                              npcs=list(self.npcs))
        return rs

    def on_received_get_robot_state(self, context: Context, data: GetRobotState):
        robot_name = data.robot_name

        rs = self._get_robot_state(robot_name)
        # timing information
        t = timestamp_from_seconds(self.current_time)
        ts = TimeSpec(time=t, frame=self.episode_name, clock=context.get_hostname())
        timing = TimingInfo(acquired={'state': ts})
        context.write('robot_state', rs, timing=timing)  # , with_schema=True)

    def on_received_dump_state(self, context: Context):
        duckiebots = {}
        for robot_name in self.pcs:
            duckiebots[robot_name] = self._get_robot_state(robot_name).state
        for robot_name in self.npcs:
            duckiebots[robot_name] = self._get_robot_state(robot_name).state
        simstate = DTSimState(t_effective=self.current_time, duckiebots=duckiebots)
        res = DTSimStateDump(simstate)
        context.write('state_dump', res)

    def on_received_get_sim_state(self, context: Context):
        done = False
        done_why = None
        done_code = None
        robot_states: Dict[str, DTSimRobotState]
        robot_states = {k: self._get_robot_state(k) for k in list(self.pcs) + list(self.npcs)}

        for robot, pc in self.pcs.items():
            state = robot_states[robot]
            q = state.state.pose
            # q, v = pc.state.TSE2_from_state()
            if self.config.terminate_on_ool:
                lprs: List[GetLanePoseResult] = list(get_lane_poses(self.dm, q))
                if not lprs:
                    msg = f'Robot {robot} is out of the lane.'
                    logger.error(msg, pc=pc)
                    done = True
                    done_why = msg
                    done_code = 'out-of-lane'
            if self.config.terminate_on_out_of_tile:
                tile_coords = is_on_a_tile(self.dm, q)
                if tile_coords is None:
                    msg = f'Robot {robot} is out of tile.'
                    logger.error(msg, pc=pc)
                    done = True
                    done_why = msg
                    done_code = 'out-of-tile'
            if self.config.terminate_on_collision:
                for other_robot, its_state in robot_states.items():
                    if other_robot == robot:
                        continue
                    q2 = its_state.state.pose
                    d = relative_pose(q2, q)
                    dist = np.linalg.norm(translation_from_O3(d))
                    if dist < self.config.collision_threshold:
                        msg = f'Robot {robot} collided with {other_robot}'
                        logger.error(msg, pc=pc)
                        done = True
                        done_why = msg
                        done_code = 'collision'

                """ Terminate on out of tile """

        sim_state = SimulationState(done, done_why, done_code)
        context.write('sim_state', sim_state)

    def on_received_get_ui_image(self, context: Context):
        profile_enabled = self.config.debug_profile
        TOPDOWN_SIZE = self.config.topdown_resolution, self.config.topdown_resolution
        if self.config.debug_no_video:
            shape = TOPDOWN_SIZE[0], TOPDOWN_SIZE[1], 3
            top_down_observation = np.zeros(shape, 'uint8')
        else:
            step = 'on_received_get_ui_image/render_top_down'
            with timeit(step, context, min_warn=0, enabled=profile_enabled):
                # noinspection PyProtectedMember
                top_down_observation = self.env._render_img(
                    TOPDOWN_SIZE[0],
                    TOPDOWN_SIZE[1],
                    self.top_down_multi_fbo,
                    self.top_down_final_fbo,
                    self.top_down_img_array,
                    top_down=True
                )

        jpg_data = rgb2jpg(top_down_observation)
        jpg = JPGImage(jpg_data)
        context.write('ui_image', jpg)


def get_snapshots(last_obs_time: float, current_time: float, until: float, dt: float) -> Iterator[float]:
    t = last_obs_time + dt
    while t < until:
        if t > current_time:
            yield t
        t += dt


def rgb2jpg(rgb: np.ndarray) -> bytes:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    compress = cv2.imencode('.jpg', bgr)[1]
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
    msg = f'timeit: {int((t1 - t0) * 1000):4d} ms for {s}'
    if delta > min_warn:
        context.info(msg)


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

    msg = ''
    msg += f'\ni, j: {i}, {j}'
    msg += f'\nPose: {geometry.SE2.friendly(q)}'
    msg += f'\nPose: {geometry.SE2.friendly(q2)}'
    msg += f'\nCur pos: {cur_pos}'
    context.info(msg)

    if tile is None:
        msg = 'Current pose is not in a tile: \n' + msg
        raise Exception(msg)

    kind = tile['kind']
    is_straight = kind.startswith('straight')

    context.info(f'Sampled tile  {tile["coords"]} {tile["kind"]} {tile["angle"]}')

    if not is_straight:
        context.info('not on a straight tile')

    # noinspection PyProtectedMember
    valid = env._valid_pose(cur_pos, cur_angle)
    context.info(f'valid: {valid}')

    try:
        lp = env.get_lane_pos2(cur_pos, cur_angle)
        context.info(f'Sampled lane pose {lp}')
        context.info(f'dist: {lp.dist}')
    except NotInLane:
        raise

    if not valid:
        msg = 'Not valid'
        context.error(msg)


def get_rgb_tuple(x: RGB) -> Tuple[float, float, float]:
    return x.r, x.g, x.b


def main():
    setup_logging()
    node = GymDuckiebotSimulator()
    protocol = protocol_simulator_DB20
    wrap_direct(node=node, protocol=protocol)


if __name__ == '__main__':
    main()
