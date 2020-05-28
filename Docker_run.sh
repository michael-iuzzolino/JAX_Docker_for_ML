export FOLDER=/home/michael/JAX_Docker_for_ML # e.g., export FOLDER=/home/michael/JAX_Docker_for_ML
export HDD_DIR=/hdd/mliuzzolino/

DEFAULT_PORT=8990
if [ -z "$1" ]
  then
    PORT=$DEFAULT_PORT
else
    PORT=$1
fi

echo "Launching on port $PORT"

docker run --runtime=nvidia -it -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
    --volume $FOLDER:/root/mount/jax \
    --mount type=bind,source=$HDD_DIR,target=/root/mount/jax/datasets/ \
    --env QT_X11_NO_MITSHM=1 \
    -p $PORT:$PORT \
    jax_env
