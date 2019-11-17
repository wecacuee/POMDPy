FROM tensorflow/tensorflow:1.14.0-gpu-py3-jupyter

ENV DATA_ROOT $HOME/dat/generative_inpainting
ENV CODE_DIR $HOME/wrk/generative_inpainting
RUN apt-get update && apt-get install -y git python3-pip && rm -rf /var/lib/apt/lists/*
RUN mkdir -p $(dirname $CODE_DIR) \
    && git clone https://github.com/wecacuee/generative_inpainting.git $CODE_DIR
RUN cd $CODE_DIR && bash setup_once.bash
WORKDIR $CODE_DIR
CMD ["bash"]

