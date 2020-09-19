ARG AIDO_REGISTRY
FROM ${AIDO_REGISTRY}/duckietown/gym-duckietown-server-python3:daffy

ARG PIP_INDEX_URL
ENV PIP_INDEX_URL=${PIP_INDEX_URL}

WORKDIR /project

COPY requirements* ./
RUN pip install -r requirements.resolved
RUN pipdeptree


COPY . .


ENTRYPOINT ["/bin/bash", "/project/launch.sh"]
