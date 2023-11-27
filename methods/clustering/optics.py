import argparse
import os

from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import KMeans
import torch
from project_utils.cluster_utils import str2bool
from project_utils.general_utils import seed_torch
from project_utils.cluster_and_log_utils import log_accs_from_preds

from methods.clustering.feature_vector_dataset import FeatureVectorDataset
from data.get_datasets import get_datasets, get_class_splits
from methods.clustering.faster_mix_k_means_pytorch import K_Means as SemiSupKMeans

from tqdm import tqdm
from config import feature_extract_dir
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS
from sklearn.manifold import TSNE
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# TODO: Debug
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


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

    # -----------------------
    # K-MEANS
    # -----------------------

    print('Collating done, starting transformation...')

    mask_lab = mask_lab.astype(bool)
    mask_cls = mask_cls.astype(bool)

    all_feats = np.concatenate(all_feats)

    l_feats = all_feats[mask_lab]       # Get labelled set
    u_feats = all_feats[~mask_lab]      # Get unlabelled set
    l_targets = targets[mask_lab]       # Get labelled targets
    u_targets = targets[~mask_lab]

    # u_targets = u_targets.cpu().numpy()

    mask = mask_cls[~mask_lab]
    mask = mask.astype(bool)       # Get unlabelled targets

    total_feats = np.concatenate((l_feats, u_feats), axis=0)
    total_targets = np.concatenate((l_targets, u_targets), axis=0)
    # print(total_feats.shape)
    # print(total_targets.shape)
    # print(np.unique(total_targets))

    # scaler = StandardScaler()
    # scaler.fit(total_feats)

    # total_feat_scaled = scaler.transform(total_feats)
    total_feat_scaled = total_feats

    # print(args.eps, args.min_samples)

    # for i in range(5):
    #     print(total_feat_scaled[i].min(), total_feat_scaled[i].max())
    #     rand_a = random.randint(0, len(total_feat_scaled)-1)
    #     a = total_feat_scaled[rand_a]
    #     rand_b = random.randint(0, len(total_feat_scaled)-1)
    #     b = total_feat_scaled[rand_b]
    #     distance = np.sqrt(np.sum((a-b)**2))
    #     print(distance)

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
    print('Current min_sample: '+str(args.min_samples))
    for min_sample_num in range(2, 10):
        for eps_delta in range(5):
            eps = args.eps+eps_delta*0.02
            dbscan = OPTICS(xi=eps, min_samples=min_sample_num)
            clusters = dbscan.fit_predict(total_feat_scaled)
            preds = clusters[~mask_lab]
            unique, count = np.unique(preds, return_counts = True)
            if(len(count)>1):
                print(min_sample_num, eps, count[0], count[1])
            else:
                print(min_sample_num, eps, count[0])
            # print(min_sample_num, eps)
            all_acc, old_acc, new_acc = log_accs_from_preds(y_true=u_targets, y_pred=preds, mask=mask, eval_funcs=args.eval_funcs,
                                                    save_name='SS-K-Means Train ACC Unlabelled', print_output=True)
            current_status = [count[0], count[1], all_acc, old_acc, new_acc]
            current_tup = (min_sample_num, eps)
            if(min_outlier[0]>count[0]):
                min_outlier = current_status
                min_outlier_tup = current_tup
            if(min_first_class[1]>count[1]):
                min_first_class = current_status
                min_first_class_tup = current_tup
            if(max_all_acc[2]<all_acc):
                max_all_acc = current_status
                max_all_acc_tup = current_tup
            if(max_old_acc[3]<old_acc):
                max_old_acc = current_status
                max_old_acc_tup = current_tup
            if(max_new_acc[4]<new_acc):
                max_new_acc = current_status
                max_new_acc_tup = current_tup
    print("Final result: ")
    print(min_outlier_tup, min_outlier)
    print(min_first_class_tup, min_first_class)
    print(max_all_acc_tup, max_all_acc)
    print(max_old_acc_tup, max_old_acc)
    print(max_new_acc_tup, max_new_acc)
            

    
    # dbscan = DBSCAN(eps=args.eps, min_samples=args.min_samples)
    # clusters = dbscan.fit_predict(total_feat_scaled)
    # print(np.unique(clusters, return_counts = True))
    # tsne_model = TSNE(2)
    # tsne_feat = tsne_model.fit_transform(total_feat_scaled)
    # print(tsne_feat.shape)
    # print(clusters.shape)
    # print(total_targets.shape)
    # # print(tsne_feat[0])
    # print(tsne_feat[:10])
    # print(clusters[:10])
    # print(total_targets[:10])

    # # First create dataframe
    # data = {'x': tsne_feat[:,0], 'y': tsne_feat[:,1], 'label': clusters}
    # df = pd.DataFrame(data)
    # # plt with dbscan cluster
    # sns.scatterplot(data=df, x='x', y='y', hue='label', palette='hls')
    # plt.title('DBSCAN result')
    # plt.legend()
    # plt.savefig('plot_dbscan.png')

    # # # clear plt
    # plt.clf()

    # # First create dataframe
    # data = {'x': tsne_feat[:,0], 'y': tsne_feat[:,1], 'label': total_targets}
    # df = pd.DataFrame(data)
    # # plt with gt cluster
    # sns.scatterplot(data=df, x='x', y='y', hue='label', palette='hls')
    # plt.title("GT result")
    # plt.legend()
    # plt.savefig('plot_gt.png')

    



    # print('Fitting Semi-Supervised K-Means...')
    # kmeans = SemiSupKMeans(k=K, tolerance=1e-4, max_iterations=args.max_kmeans_iter, init='k-means++',
    #                        n_init=args.k_means_init, random_state=None, n_jobs=None, pairwise_batch_size=1024, mode=None)

    # l_feats, u_feats, l_targets, u_targets = (torch.from_numpy(x).to(device) for
    #                                           x in (l_feats, u_feats, l_targets, u_targets))

    # kmeans.fit_mix(u_feats, l_feats, l_targets)
    # all_preds = kmeans.labels_.cpu().numpy()
    # u_targets = u_targets.cpu().numpy()

    # # -----------------------
    # # EVALUATE
    # # -----------------------
    # # Get preds corresponding to unlabelled set
    # preds = all_preds[~mask_lab]

    # # Get portion of mask_cls which corresponds to the unlabelled set
    # mask = mask_cls[~mask_lab]
    # mask = mask.astype(bool)

    # # -----------------------
    # # EVALUATE
    # # -----------------------
    # all_acc, old_acc, new_acc = log_accs_from_preds(y_true=u_targets, y_pred=preds, mask=mask, eval_funcs=args.eval_funcs,
    #                                                 save_name='SS-K-Means Train ACC Unlabelled', print_output=True)

    return all_acc, old_acc, new_acc, kmeans


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
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
    parser.add_argument('--eps', type=float, default = 0.03)
    parser.add_argument('--min_samples', type =int, default = 10 )

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    cluster_accs = {}
    seed_torch(0)
    args.save_dir = os.path.join(args.root_dir, f'{args.model_name}_{args.dataset_name}')

    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    device = torch.device('cuda:0')
    args.device = device
    print(args)

    if args.warmup_model_exp_id is not None:

        args.save_dir += '_' + args.warmup_model_exp_id

        if args.use_best_model:
            args.save_dir += '_best'

        print(f'Using features from experiment: {args.warmup_model_exp_id}')
    else:
        print(f'Using pretrained {args.model_name} features...')

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
    all_acc, old_acc, new_acc, kmeans = test_kmeans_semi_sup(train_loader, args, K=args.K)
    cluster_save_path = os.path.join(args.save_dir, 'ss_kmeans_cluster_centres.pt')
    torch.save(kmeans.cluster_centers_, cluster_save_path)