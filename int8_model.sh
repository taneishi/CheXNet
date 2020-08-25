# int8 model
source /opt/intel/openvino/bin/setupvars.sh

mkdir -p annotations

python annotation.py chest_xray --annotation_file labels/val_list.txt -ss 200 \
    -o annotations -a chestx.pickle -m chestx.json --data_dir images

python calibrate.py --config config/chestx.yml -d config/def.yml \
    -M ${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer \
    --models model --annotations annotations --batch_size 64

python ov-inference.py --int8
