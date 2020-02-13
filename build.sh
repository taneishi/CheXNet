INSTALL_DIR=/opt/intel/openvino

source ${INSTALL_DIR}/bin/setupvars.sh

cd /CheXNet

python3 ${INSTALL_DIR}/deployment_tools/model_optimizer/mo_onnx.py --input_model ${PWD}/model/densenet121.onnx --data_type=FP32 --batch 1

python3 annotation.py chest_xray --annotation_file ChestX-ray14/labels/val_list.txt -ss 200 -o annotations \
    -a chestx.pickle -m chestx.json --data_dir ChestX-ray14

python3 calibrate.py --config chestx.yml -d def.yml -M ${INSTALL_DIR}/deployment_tools/model_optimizer \
    --models . --annotations annotations --batch_size 64
