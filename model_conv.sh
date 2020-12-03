#!/bin/bash
#PBS -l nodes=1:ppn=2
#PBS -N chexnet_conv
#PBS -j oe
#PBS -o log/output_conv.log

cd ${PBS_O_WORKDIR}
mkdir -p model log

# l_openvino_toolkit_p_2019.3.376.tgz
source ${PBS_O_HOME}/intel/openvino/bin/setupvars.sh

# fp32 model optimization
python ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/mo_onnx.py \
    --input_model ${PWD}/model/densenet121.onnx --data_type=FP32 --batch 1 --output_dir model

mkdir -p annotations

source activate pytorch

python annotation.py chest_xray --annotation_file labels/val_list.txt -ss 200 \
    -o annotations -a chestx.pickle -m chestx.json --data_dir images

# int8 quantization
python calibrate.py --config config/chestx.yml -d config/def.yml \
    -M ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer \
    --models model --annotations annotations --batch_size 64
