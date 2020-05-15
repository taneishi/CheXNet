# int8 model

mkdir -p annotations

python3 annotation.py chest_xray --annotation_file ChestX-ray14/labels/val_list.txt -ss 200 \
    -o annotations -a chestx.pickle -m chestx.json --data_dir ChestX-ray14

python3 calibrate.py --config config/chestx.yml -d config/def.yml \
    -M ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer \
    --models model --annotations annotations --batch_size 64

python3 ov-inference.py int8
