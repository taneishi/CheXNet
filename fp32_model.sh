# fp32 model

python3 ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/mo_onnx.py \
    --input_model ${PWD}/model/densenet121.onnx --data_type=FP32 --batch 1 --output_dir model

python3 ov-inference.py fp32
