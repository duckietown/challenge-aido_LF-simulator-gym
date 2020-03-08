from aido_schemas import protocol_simulator
from zuper_nodes_wrapper.wrapper_outside import ComponentInterface


sim_in, sim_out = ...

sim_ci = ComponentInterface(sim_in, sim_out,
                                expect_protocol=protocol_simulator, nickname="simulator")
