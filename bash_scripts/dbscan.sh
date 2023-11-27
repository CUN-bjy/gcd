# PYTHON='/users/sagar/miniconda3/envs/open_world_prototype/bin/python'

hostname
nvidia-smi

export CUDA_VISIBLE_DEVICES=0

# Get unique log file,
SAVE_DIR=./save_dir/

EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo $EXP_NUM

python -m methods.clustering.dbscan --dataset_name 'cifar100' --semi_sup 'True' --use_ssb_splits 'False' \
 --use_best_model 'True' --max_kmeans_iter 200 --k_means_init 100 --warmup_model_exp_id '(13.11.2023_|_52.535)'\
 > ${SAVE_DIR}logfile_${EXP_NUM}.out