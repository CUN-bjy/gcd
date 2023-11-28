# PYTHON='/users/sagar/miniconda3/envs/open_world_prototype/bin/python'

export CUDA_VISIBLE_DEVICES=5

# python -m methods.clustering.extract_features --dataset cifar100 --use_best_model 'False' \
#  --warmup_model_dir './save_dir/metric_learn_gcd/log/(13.11.2023_|_52.535)/checkpoints/model.pt'

python -m methods.clustering.extract_features --dataset imagenet_100 --use_best_model 'True'