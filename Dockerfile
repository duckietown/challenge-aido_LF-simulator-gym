ARG AIDO_REGISTRY
FROM ${AIDO_REGISTRY}/duckietown/gym-duckietown-server-python3:daffy

ARG PIP_INDEX_URL
ENV PIP_INDEX_URL=${PIP_INDEX_URL}

WORKDIR /project

RUN apt-get update && apt-get install -y gcc
COPY requirements* ./
RUN pip install -U pip>=20.2
RUN pip install --use-feature=2020-resolver -r requirements.resolved
RUN pipdeptree


COPY . .


ENTRYPOINT ["/bin/bash", "/project/launch.sh"]
