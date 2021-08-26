# %% RFS Energy anomaly detection of Defect using Multivariate Gaussian of normal samples
# by Ammar Kamoona @2021

import numpy as np
import numpy
import torch
import matplotlib.pyplot as plt
import pandas as pd
from mvtec_train import PossionMLE
from utils.mvtec_utils import get_mvtec
from utils import Possion_UnnorLoglike
from sklearn.metrics import roc_auc_score, confusion_matrix, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support as prf
from torchvision import transforms as T
import torchvision
from PIL import Image
import cv2
from sklearn import metrics
generator = torch.Generator().manual_seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
import yaml
import argparse
import os
from sklearn.metrics import roc_auc_score

# %%

## iterate over objects
auc_mean_dataset_mah = []
auc_mean_dataset_rfs_energy = []
auc_mean_dataset_rfs_like = []
object_names_list = []


def eval_MHD(model, dataloaders, device, args, lamda_hat, card_mle):
    """Testing RFS energy model"""

    dataloader_train, testloader_set, card_trainer, testloader_set, trainloader_eval = dataloaders
    train_mu, train_cov = model
    train_mu = torch.tensor(train_mu, dtype=float).to(device)
    train_cov = torch.tensor(train_cov, dtype=float).to(device)
    print('Testing...')

    # Obtaining Labels and energy scores for train data

    from torch.distributions import multivariate_normal
    gmm = multivariate_normal.MultivariateNormal(train_mu, train_cov)
    from scipy.spatial.distance import mahalanobis

    Mahal_test = []
    RFS_energy_test = []
    RFS_like_test= []
    labels_test = []

    args.pick_topk = 0.03
    for batch in testloader_set:
        x = batch[0]
        y = batch[1]
        card = batch[2]

        x = x.float().to(device).squeeze()
        cov_inv = np.linalg.inv(train_cov.detach().cpu().numpy())
        dist = torch.tensor(
            [mahalanobis(sample.cpu().numpy(), train_mu.cpu().numpy(), cov_inv) ** 2 for sample in x]).to(device)
        Maha_dist = torch.tensor(
            [mahalanobis(sample.cpu().numpy(), train_mu.cpu().numpy(), cov_inv) for sample in x]).to(device)
        samples_log_like = gmm.log_prob(x)
        # dist = torch.log(torch.tensor([mahalanobis(sample.cpu().numpy(), train_mu.cpu().numpy(), cov_inv) for sample in x]).to(device))
        if args.rank_mahl:
            pick_topk = args.pick_topk
            topk = torch.round(card * pick_topk)
            top_dist = torch.topk(dist, int(topk.numpy()))
            dist = top_dist[0]
            # card=torch.tensor(dist.shape)
        sample_Mahal = Maha_dist
        sample_energy = 0.5 * dist
        card_loss_like = card_mle.compute_loss(lamda_hat, card.float().to(device), with_NLL=False)
        # card_loss2 = card_mle.compute_loss(lamda_hat, card.float().to(device), with_NLL=True)
        # new card_loss calculation using unnormlaized likeliihood
        card_loss = Possion_UnnorLoglike(lamda_hat, card.float().to(device), with_NLL=False)
        neg_card_loss = Possion_UnnorLoglike(lamda_hat, card.float().to(device), with_NLL=True)

        # sum the loglilihood of all smaples based on NAive log
        if 1:


            samples_Mahal = torch.sum(sample_Mahal, dim=0).view(-1)
            RFS_set_energy = torch.sum(sample_energy, dim=0).view(-1) + neg_card_loss.view(-1)
            RFS_set_like = torch.sum(samples_log_like, dim=0).view(-1) + card_loss_like.view(-1) + (
                            card.float().to(device=device) + 1).lgamma().view(-1)
            card_sample_test = card_loss + (card.float().to(device=device) + 1).lgamma()


        Mahal_test.append(samples_Mahal.detach().cpu())
        RFS_energy_test.append(RFS_set_energy.detach().cpu())
        labels_test.append(y)
        RFS_like_test.append(RFS_set_like.detach().cpu())

    Mahal_test = torch.cat(Mahal_test).numpy()
    RFS_energy_test = torch.cat(RFS_energy_test).numpy()
    labels_test = torch.cat(labels_test).numpy()
    RFS_like_test = torch.cat(RFS_like_test).numpy()

    auc_thre = 0

    auc_mahal = roc_auc_score(labels_test, Mahal_test) * 100
    auc_RFS_energy = roc_auc_score(labels_test, RFS_energy_test) * 100
    auc_RFS_like = roc_auc_score(labels_test, RFS_like_test) * 100
    # ploting the ROC AUC curve
    fpr, tpr, thresholds = roc_curve(labels_test, RFS_energy_test, pos_label=1)

    # display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=np.round(auc_score2, 1), estimator_name=args.object)
    # display.plot()
    # plt.show()
    # plt.savefig(args.object)

    print('ROC AUC score using sum of Mahalanobis distance : {:.2f}%'.format(auc_mahal))
    print('ROC AUC score using RFS energy: {:.2f}%'.format(auc_RFS_energy))
    print('ROC AUC score using RFS loglikelihood : {:.2f}%'.format(auc_RFS_like))

    return auc_mahal, auc_RFS_energy , auc_RFS_like, fpr, tpr


def Train_MHVD(args, data, device):

    train_loader = data
    for i, batch in enumerate(train_loader):
        input = batch[0]
        x = input
        x = x.float()
        from sklearn.covariance import LedoitWolf
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        # pca.fit(x)
        # xx=pca.transform(x)
        # xx=torch.tensor(xx)
        mean = torch.mean(x, dim=0).detach().cpu().numpy()
        # covariance estimation by using the Ledoit. Wolf et al. method
        cov = LedoitWolf().fit(x).covariance_

    return [mean, cov]


ax = plt.gca()

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--d2-net-features-path', type=str, default='./d2-net-features/',
                    help='d2-net-features file path:')
parser.add_argument('--feat_type', type=str, default='d2_net',help='local/point pattern features types')
parser.add_argument('--batch_size', type=str, default='full')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--rank_mahl', type=int, default=1)
parser.add_argument('--fewshots', type=int, default=0)
parser.add_argument('--fewshots_exm', type=int, default=15)
parser.add_argument('--use_batch_norm', type=int, default=1)
parser.add_argument('--normalise_data', type=int, default=1)


args = parser.parse_args()
for obj in range(0, 15):

    if obj == 0:
        config = './configs/mvtec_bottle_config.yaml'
        obj_name = 'bottle'


    if obj == 1:
        config = './configs/mvtec_cable_config.yaml'
        obj_name = 'cable'


    if obj == 2:
        config = './configs/mvtec_capsule_config.yaml'
        obj_name = 'capsule'


    if obj == 3:
        config = './configs/mvtec_hazelnut_config.yaml'
        obj_name = 'hazelnut'

    if obj == 4:
        config = './configs/mvtec_metal_nut_config.yaml'
        obj_name = 'metal_nut'

    if obj == 5:
        config = './configs/mvtec_pill_config.yaml'
        obj_name = 'pill'

    if obj == 6:
        config = './configs/mvtec_screw_config.yaml'
        obj_name = 'screw'

    if obj == 7:
        config = './configs/mvtec_toothbrush_config.yaml'
        obj_name = 'toothbrush'

    if obj == 8:
        config = './configs/mvtec_transistor_config.yaml'
        obj_name = 'transistor'

    if obj == 9:
        config = './configs/mvtec_zipper_config.yaml'
        obj_name = 'zipper'

    if obj == 10:
        config = './configs/mvtec_carpet_config.yaml'
        obj_name = 'carpet'

    if obj == 11:
        config = './configs/mvtec_grid_config.yaml'
        obj_name = 'grid'

    if obj == 12:
        config = './configs/mvtec_leather_config.yaml'
        obj_name = 'leather'

    if obj == 13:
        config = './configs/mvtec_tile_config.yaml'
        obj_name = 'tile'

    if obj == 14:
        config = './configs/mvtec_wood_config.yaml'
        obj_name = 'wood'


    object_names_list.append(obj_name)





    with open(config, 'r') as f:
        config = yaml.safe_load(f)

    args.object = config['dataset']['object']
    if args.feat_type == 'lf_net':
        args.data_dir = config['dataset']['data_dir_lf_net']
        args.input_dim = 256
    if args.feat_type == "d2_net":
        args.data_dir = config['dataset']['data_dir_d2_net']
        args.input_dim = 512
    if args.feat_type == "r2d2":
        args.data_dir = config['dataset']['data_dir_r2d2']
        input_dim = 128
    if args.feat_type == "sp-sift":
        args.data_dir = config['dataset']['sp']['data_dir_sift_desc']
        args.input_dim = 128
    if args.feat_type == "sp":
        args.data_dir = config['dataset']['sp']['data_dir_sp']
        args.input_dim = 256

    print('================================================================')
    print('Running for object:__', obj_name, '__')
    print('feature type', args.feat_type)
    if args.rank_mahl == 0:
        args.pick_topk = False

    if args.fewshots:
        print('fewshot mode with', str(args.fewshots_exm))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load the data
    data = get_mvtec(args, with_vaild=False)
    cardnality_loader = data[1]
    if args.fewshots:
        err, err, err, err, data = get_mvtec(args, train_only=False, with_vaild=False)
        data_new = []
        for i, batch in enumerate(data):
            input = batch[0].squeeze()
            data_new.append(input)
            if i == args.fewshots_exm:
                break
        data_new = torch.cat(data_new).to(device)
        data = data_new

    print('batch_size:', args.batch_size)


    if args.fewshots:
        from sklearn.covariance import LedoitWolf

        mean = torch.mean(data, dim=0).detach().cpu().numpy()
        # covariance estimation by using the Ledoit. Wolf et al. method
        cov = LedoitWolf().fit(data.detach().cpu().numpy()).covariance_
        mean_cov = [mean, cov]

    else:
        mean_cov = Train_MHVD(args, data[0], device)

    card_mle = PossionMLE(args, cardnality_loader, device)
    lamda_hat = card_mle.comput_lamda()
    data = get_mvtec(args, train_only=False, with_vaild=False)

    auc_score_mah, auc_score_rfs_energy, auc_score_rfs_likelihood, fpr, tpr = eval_MHD(mean_cov, data, device, args,
                                                                       lamda_hat, card_mle)


    auc_score_rfs_energy = np.round(auc_score_rfs_energy, 1)

    display1 = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc_score_rfs_energy, estimator_name=args.object)

    display1.plot(ax=ax)

# plt.plot(fpr,tpr)


    auc_mean_dataset_mah.append(auc_score_mah)
    auc_mean_dataset_rfs_energy.append(auc_score_rfs_energy)
    auc_mean_dataset_rfs_like.append(auc_score_rfs_likelihood)

    torch.cuda.empty_cache()
plt.show()
# plt.savefig('roc_3.svg', format='svg',dpi=300)
# %%
auc_mean_dataset_mah = torch.tensor(auc_mean_dataset_mah )
auc_mean_dataset_rfs_energy = torch.tensor(auc_mean_dataset_rfs_energy)
auc_mean_dataset_rfs_like = torch.tensor(auc_mean_dataset_rfs_like)


print('====================Mean ROCAUC of mvtec dataset objects================')
print('avg of AUC Sum of Mahalanobis distances {:0.2f}'.format(auc_mean_dataset_mah .mean().numpy()))
print('avg of AUC RFS log-likelihood is {:0.2f}'.format(auc_mean_dataset_rfs_like.float().mean().numpy()))
print('avg of AUC RFS Energy {:0.2f}'.format(auc_mean_dataset_rfs_energy .mean().numpy()))




