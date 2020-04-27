FROM ubuntu:16.04

ARG INSTALL_DIR=/opt/intel/openvino
ARG TEMP_DIR=/tmp/openvino_installer

RUN apt-get update && apt-get install -y --no-install-recommends \
    cpio \
    bzip2 \
    sudo \
    lsb-release && \
    rm -rf /var/lib/apt/lists/*

COPY l_openvino_toolkit_p_2019.3.376.tgz /tmp/
#COPY l_openvino_toolkit_p_2020.2.120.tgz /tmp/

RUN mkdir -p $TEMP_DIR && cd $TEMP_DIR && \
    tar xf /tmp/l_openvino_toolkit*.tgz && \
    cd l_openvino_toolkit* && \
    sed -i 's/decline/accept/g' silent.cfg && \
    ./install.sh -s silent.cfg && \
    rm -rf $TEMP_DIR

RUN cd $INSTALL_DIR/deployment_tools/model_optimizer/install_prerequisites && \
    bash install_prerequisites_onnx.sh

RUN cd $INSTALL_DIR/install_dependencies && \
    bash install_openvino_dependencies.sh

RUN pip3 install torch torchvision scikit-learn shapely yamlloader nibabel tqdm xmltodict progress py-cpuinfo

# docker run -it --rm --ipc=host -m 64gb -v ${PWD}:/CheXNet openvino
