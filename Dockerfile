ARG AIDO_REGISTRY
FROM ${AIDO_REGISTRY}/duckietown/gym-duckietown:daffy-amd64

ARG PIP_INDEX_URL
ENV PIP_INDEX_URL=${PIP_INDEX_URL}

WORKDIR /project

RUN apt-get update && apt-get install -y gcc

RUN apt-get install -y xauth

RUN pip install -U "pip>=20.2"
COPY requirements.* ./
RUN cat requirements.* > .requirements.txt
RUN  pip3 install --use-feature=2020-resolver -r .requirements.txt

RUN pip list
RUN pipdeptree




COPY . .

RUN pip install --no-deps .

RUN python3 -c "from duckietown_simulator_gym import *"

# needs X
#RUN pwd && ls && PYTHON_PATH=. python3 -c "import gym_bridge"

ENTRYPOINT ["/bin/bash", "/project/launch.sh"]
