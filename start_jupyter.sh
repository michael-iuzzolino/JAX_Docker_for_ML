#!/bin/bash
. activate jax

DEFAULT_PORT=8990
if [ -z "$1" ]
  then
    PORT=$DEFAULT_PORT
else
    PORT=$1
fi

jupyter notebook --ip 0.0.0.0 --port $PORT --no-browser --allow-root
