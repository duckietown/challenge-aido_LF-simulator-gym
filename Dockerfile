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
RUN pip3 install  -r .requirements.txt
RUN pip uninstall dataclasses -y
RUN pip list
#RUN pipdeptree # will fail

COPY . .

RUN pip install --no-deps .

RUN node-launch --config node_launch.yaml --check

ENTRYPOINT ["/bin/bash", "/project/launch.sh"]
