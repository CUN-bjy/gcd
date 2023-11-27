# PYTHON='/users/sagar/miniconda3/envs/open_world_prototype/bin/python'

hostname
nvidia-smi

export CUDA_VISIBLE_DEVICES=0
# Get unique log file
SAVE_DIR=./save_dir/

EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo $EXP_NUM

python -m methods.estimate_k.estimate_k --max_classes 300 --dataset_name cifar100 --search_mode other \
        > ${SAVE_DIR}logfile_${EXP_NUM}.out