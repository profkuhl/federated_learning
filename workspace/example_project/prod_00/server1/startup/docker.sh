#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# docker run script for FL server
# to use host network, use line below
NETARG="--net=host"
# or to expose specific ports, use line below
#NETARG="-p 8003:8003 -p 8002:8002"
DOCKER_IMAGE=localhost/nvflare:0.0.1
echo "Starting docker with $DOCKER_IMAGE"
svr_name="${SVR_NAME:-flserver}"
mode="${1:-r}"
if [ $mode = "-d" ]
then
  docker run -d --rm --name=$svr_name -v $DIR/..:/workspace/ -w /workspace \
  --ipc=host $NETARG $DOCKER_IMAGE /bin/bash -c \
  "python -u -m nvflare.private.fed.app.server.server_train -m /workspace -s fed_server.json --set secure_train=true config_folder=config org=nvidia"
else
  docker run --rm -it --name=$svr_name -v $DIR/..:/workspace/ -w /workspace/ --ipc=host $NETARG $DOCKER_IMAGE /bin/bash
fi
