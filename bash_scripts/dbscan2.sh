# PYTHON='/users/sagar/miniconda3/envs/open_world_prototype/bin/python'

export CUDA_VISIBLE_DEVICES=5

# Get unique log file,
SAVE_DIR=./save_dir/

# EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
# EXP_NUM=$((${EXP_NUM}+1))
# echo $EXP_NUM

# python -m methods.clustering.dbscan --dataset_name 'cifar100' --semi_sup 'True' --use_ssb_splits 'False' \
#  --use_best_model 'True' --root_dir /home/junyeob/gcd/feat_dir/gcd_pretrain_cifar100 \
#  > ${SAVE_DIR}logfile_${EXP_NUM}.out

 
# EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
# EXP_NUM=$((${EXP_NUM}+1))
# echo $EXP_NUM
# python -m methods.clustering.dbscan --dataset_name 'cifar100' --semi_sup 'False' --use_ssb_splits 'False' \
#  --use_best_model 'True' --root_dir /home/junyeob/gcd/feat_dir/gcd_pretrain_cifar100 \
#  > ${SAVE_DIR}logfile_${EXP_NUM}.out

 
EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo $EXP_NUM
python -m methods.clustering.dbscan --dataset_name 'cifar100' --semi_sup 'True' --use_ssb_splits 'False' \
 --use_best_model 'True' --metric "cosine" --root_dir /home/junyeob/gcd/feat_dir/gcd_pretrain_cifar100 \
 > ${SAVE_DIR}logfile_${EXP_NUM}.out

 
EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo $EXP_NUM
python -m methods.clustering.dbscan --dataset_name 'cifar100' --semi_sup 'False' --use_ssb_splits 'False' \
 --use_best_model 'True' --metric "cosine" --root_dir /home/junyeob/gcd/feat_dir/gcd_pretrain_cifar100 \
 > ${SAVE_DIR}logfile_${EXP_NUM}.out