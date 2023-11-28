DATASET_DIR="/home/junyeob/gcd/dataset"
# -----------------
# DATASET ROOTS
# -----------------
cifar_10_root = f'{DATASET_DIR}/cifar10'
cifar_100_root = f'{DATASET_DIR}/cifar100'
cub_root = f'{DATASET_DIR}/cub'
aircraft_root = f'{DATASET_DIR}/fgvc-aircraft-2013b'
car_root = f'{DATASET_DIR}/cars'
herbarium_dataroot = f'{DATASET_DIR}/herbarium_19'
imagenet_root = f'{DATASET_DIR}/ImageNet'

# OSR Split dir
osr_split_dir = 'data/ssb_splits'

# -----------------
# OTHER PATHS
# -----------------
dino_pretrain_path = './download_resources/checkpoints/dino_vitbase16_pretrain.pth'

feature_extract_dir = './feat_dir/'     # Extract features to this directory

exp_root = 'dev_outputs' # All logs and checkpoints will be saved here