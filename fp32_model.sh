#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -N chexnet_fp32
#PBS -j oe
#PBS -o output_fp32.log

cd ${PBS_O_WORKDIR}
mkdir -p model

# fp32 model
source /opt/intel/openvino/bin/setupvars.sh

python ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/mo_onnx.py \
    --input_model ${PWD}/model/densenet121.onnx --data_type=FP32 --batch 1 --output_dir model

python ov-inference.py --fp32
