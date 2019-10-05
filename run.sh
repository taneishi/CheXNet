export OMP_SHEDULE=static
export KMP_AFFINITY=granularity=fine,compact,1,0,verbose
export MKL_NUM_THREADS=$(nproc --all)
export OMP_NUM_THREADS=$(nproc --all)
export KMP_SETTINGS=1
export KMP_BLOCKTIME=1
python model.py
