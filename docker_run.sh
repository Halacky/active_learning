docker run --gpus all -ti --ipc host --volume /home/kirill/project/active_learning:/home/active_learning --network host --name active_learning nvidia/cuda:12.0.1-cudnn8-devel-ubuntu22.04