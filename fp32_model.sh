#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -N chexnet_fp32
#PBS -j oe
#PBS -o log/output_fp32.log

cd ${PBS_O_WORKDIR}
mkdir -p model log

# l_openvino_toolkit_p_2019.3.376.tgz
source ${PBS_O_HOME}/intel/openvino/bin/setupvars.sh

# fp32 model optimization
python ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/mo_onnx.py \
    --input_model ${PWD}/model/densenet121.onnx --data_type=FP32 --batch 1 --output_dir model

python ov-inference.py --fp32
python ov-inf-async.py --fp32
