#!/usr/bin/env python3


from zuper_commons.logs import setup_logging
from zuper_nodes_wrapper import wrap_direct

from aido_schemas import (protocol_simulator_DB20)
from duckietown_simulator_gym.code import GymDuckiebotSimulator


def main():
    setup_logging()
    node = GymDuckiebotSimulator()
    protocol = protocol_simulator_DB20
    wrap_direct(node=node, protocol=protocol)


if __name__ == '__main__':
    main()
