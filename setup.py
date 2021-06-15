from setuptools import setup


def get_version(filename: str):
    import ast

    version = None
    with open(filename) as f:
        for line in f:
            if line.startswith("__version__"):
                version = ast.parse(line).body[0].value.s
                break
        else:
            raise ValueError("No version found in %r." % filename)
    if version is None:
        raise ValueError(filename)
    return version


version = get_version(filename="src/duckietown_simulator_gym/__init__.py")

line = "daffy"
install_requires = [
    "duckietown-gym-daffy",
    "aido-agents-daffy",
    "aido-protocols-daffy",
    "PyGeometry-z6",
    "zuper-nodes-z6",
    "zuper-commons-z6",
    "opencv-python",
    "PyYAML",
    "numpy",
    "duckietown-world-daffy",
]

setup(
    name=f"duckietown-simulator-gym-{line}",
    version=version,
    keywords="",
    package_dir={"": "src"},
    packages=["duckietown_simulator_gym"],
    install_requires=install_requires,
    entry_points={
        "console_scripts": [],
    },
)
