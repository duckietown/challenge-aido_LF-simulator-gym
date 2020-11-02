ARG AIDO_REGISTRY
FROM ${AIDO_REGISTRY}/duckietown/gym-duckietown-server-python3:daffy

ARG PIP_INDEX_URL
ENV PIP_INDEX_URL=${PIP_INDEX_URL}

WORKDIR /project

RUN apt-get update && apt-get install -y gcc

RUN pip install -U "pip>=20.2"
COPY requirements.* ./
RUN cat requirements.* > .requirements.txt
RUN  pip3 install --use-feature=2020-resolver -r .requirements.txt

RUN pip list
RUN pipdeptree




COPY . .

# needs X
#RUN pwd && ls && PYTHON_PATH=. python3 -c "import gym_bridge"

ENTRYPOINT ["/bin/bash", "/project/launch.sh"]
