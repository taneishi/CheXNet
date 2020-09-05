#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -N chexnet_int8
#PBS -j oe
#PBS -o log/output_int8.log

cd ${PBS_O_WORKDIR}
mkdir -p model log

source ${PBS_O_HOME}/intel/openvino/bin/setupvars.sh

python ov-inference.py --int8
python ov-inf-async.py --int8
