import argparse
import os

from torch.utils.data import DataLoader
import numpy as np
import torch
from project_utils.cluster_utils import str2bool
from project_utils.general_utils import seed_torch
from project_utils.cluster_and_log_utils import log_accs_from_preds

from methods.clustering.feature_vector_dataset import FeatureVectorDataset
from data.get_datasets import get_datasets, get_class_splits

from tqdm import tqdm
from config import feature_extract_dir
from sklearn.cluster import DBSCAN
import umap
from sklearn.metrics import pairwise_distances

# TODO: Debug
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def postprocessing(preds, features, metric):
    outliers = (preds==-1).nonzero()[0]
    if len(outliers) > 10000:
        return preds
    centroids = []
    for i in range(preds.max()+1):
        if len((preds==i).nonzero()[0]) == 0:
            centroids.append(np.ones_like(features[0])*99999)
        else:
            centroids.append(features[(preds==i).nonzero()[0]].mean(0))
        
    for ol in outliers:
        dist = [pairwise_distances(features[ol].reshape(1,-1), mean.reshape(1,-1), metric=metric) for mean in centroids]
        preds[ol] = np.argmin(dist)
    return preds

def test_kmeans_semi_sup(merge_test_loader, args, K=None):

    """
    In this case, the test loader needs to have the labelled and unlabelled subsets of the training data
    """

    if K is None:
        K = args.num_labeled_classes + args.num_unlabeled_classes

    all_feats = []
    targets = np.array([])
    mask_lab = np.array([])     # From all the data, which instances belong to the labelled set
    mask_cls = np.array([])     # From all the data, which instances belong to Old classes

    print('Collating features...')
    # First extract all features
    for batch_idx, (feats, label, _, mask_lab_) in enumerate(tqdm(merge_test_loader)):

        feats = feats.to(device)

        feats = torch.nn.functional.normalize(feats, dim=-1)

        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        mask_cls = np.append(mask_cls, np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))
        mask_lab = np.append(mask_lab, mask_lab_.cpu().bool().numpy())


    print('Collating done, starting transformation...')

    mask_lab = mask_lab.astype(bool)
    mask_cls = mask_cls.astype(bool)

    all_feats = np.concatenate(all_feats)

    u_targets = targets[~mask_lab]


    mask = mask_cls[~mask_lab]
    mask = mask.astype(bool)       # Get unlabelled targets
    
    masked_target = targets[:]
    masked_target[~mask_lab] = -1
    # embedding = umap.UMAP()
    if args.semi_sup:
        print("Semi-Supervised UMAP")
        embedding = umap.UMAP(n_components=128,output_metric='hyperboloid').fit_transform(all_feats, y=masked_target)
    else:
        print("Unsupervised UMAP")
        embedding = umap.UMAP(n_components=128,output_metric='hyperboloid').fit_transform(all_feats)
    # total_feat_scaled = scaler.transform(total_feats)
    # total_feat_scaled = all_feats
    total_feat_scaled = embedding

    min_outlier = [50000, 50000, 0, 0, 0]
    min_outlier_tup = (0,0)
    min_first_class = [50000, 50000, 0, 0, 0]
    min_first_class_tup = (0,0)
    max_all_acc = [50000, 50000, 0, 0, 0]
    max_all_acc_tup = (0,0)
    max_old_acc = [50000, 50000, 0, 0, 0]
    max_old_acc_tup = (0,0)
    max_new_acc = [50000, 50000, 0, 0, 0]
    max_new_acc_tup = (0,0)

    print('Starting DBSCAN... with various eps')
    
    metric = args.metric
    for min_sample_num in range(5, 30):
        for eps_delta in range(20):
            eps = args.eps+eps_delta*1e-7
            dbscan = DBSCAN(eps=eps, min_samples=min_sample_num, metric=metric)
            clusters = dbscan.fit_predict(total_feat_scaled)
            preds = clusters[~mask_lab]
            captured_K = clusters.max() + 1
            num_outliers = len((preds==-1).nonzero()[0])
            print(f"min_sample_num: {min_sample_num}, eps: {eps}, num_outliers: {num_outliers}, K: {captured_K}")
            all_acc, old_acc, new_acc = log_accs_from_preds(y_true=u_targets, y_pred=preds, mask=mask, eval_funcs=args.eval_funcs,
                                                    save_name='Train ACC Unlabelled', print_output=True)
            
            current_status = [captured_K, num_outliers, all_acc, old_acc, new_acc]
            current_tup = (min_sample_num, eps)
            if(min_outlier[0]>num_outliers):
                min_outlier = current_status
                min_outlier_tup = current_tup
            if(max_all_acc[2]<all_acc):
                max_all_acc = current_status
                max_all_acc_tup = current_tup
            if(max_old_acc[3]<old_acc):
                max_old_acc = current_status
                max_old_acc_tup = current_tup
            if(max_new_acc[4]<new_acc):
                max_new_acc = current_status
                max_new_acc_tup = current_tup
            
            # # postprecessing
            # post_preds = postprocessing(preds, total_feat_scaled, metric)
            # all_acc, old_acc, new_acc = log_accs_from_preds(y_true=u_targets, y_pred=post_preds, mask=mask, eval_funcs=args.eval_funcs,
            #                                         save_name='(post)Train ACC Unlabelled', print_output=True)
    print("Final result: ")
    print(min_outlier_tup, min_outlier)
    print(max_all_acc_tup, max_all_acc)
    print(max_old_acc_tup, max_old_acc)
    print(max_new_acc_tup, max_new_acc)
            
    return all_acc, old_acc, new_acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument("--metric", default="euclidean", type=str)
    parser.add_argument('--K', default=None, type=int, help='Set manually to run with custom K')
    parser.add_argument('--root_dir', type=str, default=feature_extract_dir)
    parser.add_argument('--warmup_model_exp_id', type=str, default=None)
    parser.add_argument('--use_best_model', type=str2bool, default=True)
    parser.add_argument('--spatial', type=str2bool, default=False)
    parser.add_argument('--semi_sup', type=str2bool, default=True)
    parser.add_argument('--max_kmeans_iter', type=int, default=10)
    parser.add_argument('--k_means_init', type=int, default=10)
    parser.add_argument('--model_name', type=str, default='vit_dino', help='Format is {model_name}_{pretrain}')
    parser.add_argument('--dataset_name', type=str, default='aircraft', help='options: cifar10, cifar100, scars')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v1', 'v2'])
    parser.add_argument('--use_ssb_splits', type=str2bool, default=True)
    parser.add_argument('--eps', type=float, default = 1e-8)
    parser.add_argument('--min_samples', type =int, default = 10 )

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    cluster_accs = {}
    seed_torch(0)
    args.save_dir = args.root_dir

    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    device = torch.device('cuda:0')
    args.device = device

    print(args)
    print(args.save_dir)

    # --------------------
    # DATASETS
    # --------------------
    print('Building datasets...')
    train_transform, test_transform = None, None
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,
                                                                             train_transform, test_transform, args)

    # Set target transforms:
    target_transform_dict = {}
    for i, cls in enumerate(list(args.train_classes) + list(args.unlabeled_classes)):
        target_transform_dict[cls] = i
    target_transform = lambda x: target_transform_dict[x]

    # Convert to feature vector dataset
    test_dataset = FeatureVectorDataset(base_dataset=test_dataset, feature_root=os.path.join(args.save_dir, 'test'))
    unlabelled_train_examples_test = FeatureVectorDataset(base_dataset=unlabelled_train_examples_test,
                                                          feature_root=os.path.join(args.save_dir, 'train'))
    train_dataset = FeatureVectorDataset(base_dataset=train_dataset, feature_root=os.path.join(args.save_dir, 'train'))
    train_dataset.target_transform = target_transform

    unlabelled_train_loader = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, num_workers=args.num_workers,
                                      batch_size=args.batch_size, shuffle=False)
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers,
                              batch_size=args.batch_size, shuffle=False)

    print('Performing SS-K-Means on all in the training data...')
    all_acc, old_acc, new_acc = test_kmeans_semi_sup(train_loader, args, K=args.K)