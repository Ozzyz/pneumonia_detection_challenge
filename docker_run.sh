NV_GPU=$1 nvidia-docker run -it --rm -v $PWD:/root tensorflow/tensorflow:latest-devel-gpu-py3 bash
