#!/usr/bin/env python3


import logging
import os

from aido_schemas import protocol_simulator_DB20_timestamps, wrap_direct
from duckietown_simulator_gym.code import GymDuckiebotSimulator


def set_loglevel():
    LEVELS = {
        "CRITICAL",
        "FATAL",
        "ERROR",
        "WARN",
        "WARNING",
        "INFO",
        "DEBUG",
        "NOTSET",
    }
    LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
    if LOGLEVEL in LEVELS:
        root_logger = logging.getLogger(None)
        root_logger.setLevel(LOGLEVEL)


def main():
    set_loglevel()
    # setup_logging()
    node = GymDuckiebotSimulator()
    protocol = protocol_simulator_DB20_timestamps
    wrap_direct(node=node, protocol=protocol)


if __name__ == "__main__":
    main()
