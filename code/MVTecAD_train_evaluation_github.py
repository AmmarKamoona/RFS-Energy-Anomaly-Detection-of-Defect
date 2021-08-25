# %% RFS Energy anomaly detection of Defect using Multivariate Gaussian of normal samples
# by Ammar Kamoona @2021

import numpy as np
import numpy
import torch
import matplotlib.pyplot as plt
import pandas as pd
from mvtec_train import PossionMLE
from utils.mvtec_utils import get_mvtec
from sklearn.metrics import roc_auc_score, confusion_matrix, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support as prf
from utils.utilis import detetect_keypoint_desc
from torchvision import transforms as T
import torchvision
from PIL import Image

import cv2

generator = torch.Generator().manual_seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
import yaml
import argparse
import os
from utils.utilis import detect_d2_net_descr
from sklearn.metrics import roc_auc_score

CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
# %%

## iterate over objects
mean_dataset_nv = []
rocauc_dataset_nv = []
rocauc2_dataset_nv = []
cardauc_dataset_nv = []

mean_dataset_rf = []
rocauc_dataset_rf = []
rocauc2_dataset_rf = []
cardauc_dataset_rf = []

mean_dataset_rk_csd = []
rocauc_dataset_rk_csd = []
rocauc2_dataset_rk_csd = []
cardauc_dataset_rk_csd = []
mean_dataset_rk_l2 = []
rocauc_dataset_rk_l2 = []
rocauc2_dataset_rk_l2 = []
cardauc_dataset_rk_l2 = []
object_names_list = []
parser = argparse.ArgumentParser()
parser.add_argument('--settingfile', type=str, default='./settings/DAGMM_v4/set_1.yaml',
                    help='setting file path:')

args2 = parser.parse_args()


##put funtions for now
def Possion_UnnorLoglike(lamda, card, with_NLL=False):
    Loglike = card * torch.log(lamda) - (card + 1).lgamma()
    # Loglike=card*torch.log(lamda)
    if not with_NLL:
        return Loglike
    if with_NLL:
        return -Loglike


def eval_MHD(model, dataloaders, device, args, with_card, with_rank, lamda_hat, card_mle):
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
    
    energy_test = []
    energy_test2 = []
    RFS_set_like=[]
    labels_test = []
    card_energy_test = []
    args.pick_topk=0.03
    for batch in testloader_set:
        x = batch[0]
        y = batch[1]
        card = batch[2]

        x = x.float().to(device).squeeze()
        cov_inv = np.linalg.inv(train_cov.detach().cpu().numpy())
        dist = torch.tensor(
            [mahalanobis(sample.cpu().numpy(), train_mu.cpu().numpy(), cov_inv) ** 2 for sample in x]).to(device)
        Maha_dist = torch.tensor([mahalanobis(sample.cpu().numpy(), train_mu.cpu().numpy(), cov_inv) for sample in x]).to(device)
        samples_log_like=gmm.log_prob(x)
        # dist = torch.log(torch.tensor([mahalanobis(sample.cpu().numpy(), train_mu.cpu().numpy(), cov_inv) for sample in x]).to(device))
        if args.rank_mahl:
            pick_topk = args.pick_topk
            topk = torch.round(card * pick_topk)
            top_dist = torch.topk(dist, int(topk.numpy()))
            dist = top_dist[0]
            # card=torch.tensor(dist.shape)
        sample_energy = Maha_dist
        sample_energy2 = 0.5 * dist
        card_loss_like = card_mle.compute_loss(lamda_hat, card.float().to(device), with_NLL=False)
        # card_loss2 = card_mle.compute_loss(lamda_hat, card.float().to(device), with_NLL=True)
        # new card_loss calculation using unnormlaized likeliihood
        card_loss = Possion_UnnorLoglike(lamda_hat, card.float().to(device), with_NLL=False)
        card_loss2 = Possion_UnnorLoglike(lamda_hat, card.float().to(device), with_NLL=True)
        # print('card_loss2',card_loss2)

        # sum the loglilihood of all smaples based on NAive log
        if with_card:
            if with_rank:
                # this for positive log
                # sample_energy=torch.sum(sample_energy,dim=0).view(-1)+card_loss+(torch.log(C)*card.float().to(device=device))
                # for negative log
                sample_energy = torch.sum(sample_energy2, dim=0).view(-1) + card_loss.view(-1) - (
                    card.float().to(device=device)).view(-1)
                set_energy2 = torch.sum(sample_energy2, dim=0).view(-1) + card_loss.view(-1) - (
                    card.float().to(device=device)).view(-1)
                # card_sample_test=card_loss - (torch.log(C) * card.float().to(device=device))
                card_sample_test = card_loss - (card.float().to(device=device))
            else:
                # sample_energy = torch.sum(sample_energy, dim=0).view(-1) + card_loss
                # this for positive log
                # sample_energy = torch.sum(sample_energy, dim=0).view(-1) + card_loss + (card.float().to(device=device) + 1).lgamma()
                # for negative log
                sample_energy = torch.sum(sample_energy, dim=0).view(-1)
                # set_energy2 = torch.sum(sample_energy2, dim=0).view(-1)+card_loss2.view(-1)-(card.float().to(device=device) + 1).lgamma().view(-1)
                set_energy2 = torch.sum(sample_energy2, dim=0).view(-1) + card_loss.view(-1)

                set_rfs_like = torch.sum(samples_log_like, dim=0).view(-1)+card_loss_like.view(-1)+(card.float().to(device=device) + 1).lgamma().view(-1)
                card_sample_test = card_loss + (card.float().to(device=device) + 1).lgamma()
        else:
            sample_energy = torch.sum(sample_energy, dim=0).view(-1)
            set_energy2 = torch.sum(sample_energy2, dim=0).view(-1)
            card_sample_test = torch.tensor([1])

        energy_test.append(sample_energy.detach().cpu())
        energy_test2.append(set_energy2.detach().cpu())
        labels_test.append(y)
        RFS_set_like.append(set_rfs_like.detach().cpu())
        card_energy_test.append(card_sample_test.detach().cpu())
    energy_test = torch.cat(energy_test).numpy()
    energy_test2 = torch.cat(energy_test2).numpy()
    labels_test = torch.cat(labels_test).numpy()
    RFS_set_like =torch.cat(RFS_set_like).numpy()

    auc_thre = 0

    card_auc = roc_auc_score(labels_test, card_energy_test) * 100
    auc_score = roc_auc_score(labels_test, energy_test) * 100
    auc_score2 = roc_auc_score(labels_test, energy_test2) * 100
    auc_RFS_like=roc_auc_score(labels_test,RFS_set_like)*100
    # ploting the ROC AUC curve
    fpr, tpr, thresholds = roc_curve(labels_test, energy_test2, pos_label=1)

    #display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=np.round(auc_score2, 1), estimator_name=args.object)
    #display.plot()
    # plt.show()
    #plt.savefig(args.object)

    print('ROC AUC score using loglik auc_score(labels_test, energy_test) : {:.2f}%'.format(auc_score))
    print('ROC AUC score using -loglik auc_score(labels_test, energy_test2) : {:.2f}%'.format(auc_score2))
    print('ROC AUC score using auc_score(labels_test, energy_test2) : {:.2f}%'.format(auc_thre))
    print('Card ROC AUC score using auc_score(labels_test, card_energy_test) : {:.2f}%'.format(card_auc))
    print('ROC AUC score using auc_score(labels_test, RFS likelihood) : {:.2f}%'.format(auc_RFS_like))

    return auc_score, auc_score2, auc_thre, card_auc,fpr,tpr


def Train_MHVD(args, data, device):
    from models.networks import DAGMM_v4
    from utils.utilis import weights_init_normal
    from torch import optim
    import torch.nn.functional as F
    # optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9)
    train_loader = data

    from sklearn.mixture import BayesianGaussianMixture
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


from torch.utils.data import Dataset
ax=plt.gca()
# remove the above functions for later use
for obj in range(10, 12):

    if obj == 0:
        config = './configs/mvtec_bottle_config.yaml'
        obj_name = 'bottle'
        n_gmm = 2
        # sp_thresh = 0.1
        sp_thresh = 0.0001
        sift_edgeThreshold = 10


    if obj == 1:
        config = './configs/mvtec_cable_config.yaml'
        obj_name = 'cable'
        n_gmm = 1
        sp_thresh = 0.03
        sift_edgeThreshold = 20


    if obj == 2:
        config = './configs/mvtec_capsule_config.yaml'
        obj_name = 'capsule'
        sp_thresh = 0.08
        sift_edgeThreshold = 20


    if obj == 3:
        config = './configs/mvtec_hazelnut_config.yaml'
        obj_name = 'hazelnut'
        sp_thresh = 0.0001
        sift_edgeThreshold = 40
    if obj == 4:
        config = './configs/mvtec_metal_nut_config.yaml'
        obj_name = 'metal_nut'
        sp_thresh = 0.3
        sift_edgeThreshold = 20
    if obj == 5:
        config = './configs/mvtec_pill_config.yaml'
        obj_name = 'pill'
        # sp_thresh = 0.2
        sp_thresh = 0.001
        sift_edgeThreshold = 20
    if obj == 6:
        config = './configs/mvtec_screw_config.yaml'
        obj_name = 'screw'
        sp_thresh = 0.29
        sift_edgeThreshold = 20
    if obj == 7:
        config = './configs/mvtec_toothbrush_config.yaml'
        obj_name = 'toothbrush'
        sp_thresh = 0.001
        sift_edgeThreshold = 20
    if obj == 8:
        config = './configs/mvtec_transistor_config.yaml'
        obj_name = 'transistor'
        sp_thresh = 0.04
        sift_edgeThreshold = 20
    if obj == 9:
        config = './configs/mvtec_zipper_config.yaml'
        obj_name = 'zipper'
        sp_thresh = 0.05
        sift_edgeThreshold = 20
    if obj == 10:
        config = './configs/mvtec_carpet_config.yaml'
        obj_name = 'carpet'
        sp_thresh = 0.001
        sift_edgeThreshold = 20
    if obj == 11:
        config = './configs/mvtec_grid_config.yaml'
        obj_name = 'grid'
        sp_thresh = 0.28
        sift_edgeThreshold = 20
    if obj == 12:
        config = './configs/mvtec_leather_config.yaml'
        obj_name = 'leather'
        sp_thresh = 0.0001
        sift_edgeThreshold = 20
    if obj == 13:
        config = './configs/mvtec_tile_config.yaml'
        obj_name = 'tile'
        # sp_thresh = 0.002
        sp_thresh = 0.0001
        sift_edgeThreshold = 20
    if obj == 14:
        config = './configs/mvtec_wood_config.yaml'
        obj_name = 'wood'
        sp_thresh = 0.1
        sift_edgeThreshold = 20

    object_names_list.append(obj_name)

    # settings_file='./settings/DAGMM/set_1.yaml'
    settings_file = args2.settingfile


    # print('Pring args2 seeting file path',args2.settingfile)
    class Args:

        with open(settings_file, 'r') as fff:
            Seting = yaml.safe_load(fff)
        DAGMM_net = Seting['DAGMM_net']
        DAGMM_2_net = Seting['DAGMM_2_net']
        GMSET_net = Seting['GMSET_net']
        save_model = Seting['save_model']
        percent = Seting['percent']
        feat_type = Seting['feat_type']
        feat_type = 'd2_net'
        num_epochs = Seting['num_epochs']
        num_epochs = 10
        patience = Seting['patience']
        use_cvs_file = Seting['use_cvs_file']
        model_name = Seting['model']['model_name']
        lr = Seting['model']['lr']
        batch_size = Seting['batch_size']
        batch_size = 'full'
        latent_dim = Seting['model']['latent_dim']
        add_to_latent = Seting['model']['add_to_latent']
        n_gmm = Seting['model']['n_gmm']  # number of Gaussain mixture componennts
        lambda_energy = Seting['model']['lambda_energy']
        lambda_cov = Seting['model']['lambda_cov']
        use_early_learning = Seting['model']['use_early_learning']
        Acti = Seting['model']['activation']
        use_l2_loss = Seting['model']['use_l2_loss']

        pick_top_k = Seting['pick_top_k']
        if Acti == "LeakyReLU":
            activation = torch.nn.LeakyReLU()
        if Acti == "Tanh":
            activation = torch.nn.Tanh()
        if Acti == "ReLU":
            activation = torch.nn.ReLU()
        lr_milestones = Seting['lr_milestones']
        gpu = Seting['gpu']
        card_hidden_sizes = Seting['card_hidden_sizes']
        with_card_MLE = Seting['with_card_MLE']

        with open(config, 'r') as f:
            config = yaml.safe_load(f)

        data_dir_cvs = config['dataset']['cvs_file_path']['data_dir']
        object = config['dataset']['object']
        if feat_type == 'lf_net':
            data_dir = config['dataset']['data_dir_lf_net']
            input_dim = 256
        if feat_type == "d2_net":
            data_dir = config['dataset']['data_dir_d2_net']
            input_dim = 512
        if feat_type == "r2d2":
            data_dir = config['dataset']['data_dir_r2d2']
            input_dim = 128
        if feat_type == "sp-sift":
            data_dir = config['dataset']['sp']['data_dir_sift_desc']
            input_dim = 128
        if feat_type == "sp":
            data_dir = config['dataset']['sp']['data_dir_sp']
            input_dim = 256
        hidden_sizes = [latent_dim, n_gmm]
        saved_path = config['model']['saved_path']
        card_saved_path = config['model']['saved_path']
        use_batch_norm = 1  # AE with batch normalization
        normalise_data = 1  # perform data normalization techniqies\
        gpu = '4'
        rank_mahl = 1
        fewshots = 0
        fewshots_exm = 15


    args = Args()

    print('Running for object:__', obj_name, '__')
    print('feature type', args.feat_type)
    if args.rank_mahl == 0:
        args.pick_topk = False


    if args.fewshots:
        print('fewshot mode with', str(args.fewshots_exm))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #load the data
    data = get_mvtec(args, with_vaild=False)
    cardnality_loader = data[1]
    if args.fewshots:
        err, err, err, err, data = get_mvtec_v2(args, train_only=False, with_vaild=False)
        data_new = []
        for i, batch in enumerate(data):
            input = batch[0].squeeze()
            data_new.append(input)
            if i == args.fewshots_exm:
                break
         data_new = torch.cat(data_new).to(device)
         data = data_new

    print('batch_size:', args.batch_size)
    from torch.utils.data import DataLoader

    print('batch_size:', args.batch_size)
    if args.DAGMM_net == 1:
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
    data = get_mvtec_v2(args, train_only=False, with_vaild=False)

    with_card = True
    with_rank = False

    title2 = '(RFS Energy)'
    print('================================================================')
    auc_score_rf, auc_score2_rf, mean_rf, card_auc_rf,fpr,tpr = eval_MHD(mean_cov, data, device, args, with_card,
                                                                         with_rank,
                                                                             lamda_hat, card_mle)
    from sklearn import metrics

    auc_score2_rf=np.round(auc_score2_rf, 1)

    display1 = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc_score2_rf, estimator_name=args.object)

    display1.plot(ax=ax)

    #plt.plot(fpr,tpr)



    mean_dataset_rf.append(mean_rf)
    rocauc_dataset_rf.append(auc_score_rf)
    rocauc2_dataset_rf.append(auc_score2_rf)
    cardauc_dataset_rf.append(card_auc_rf)
        # labels_rfs, scores_rfs = eval_rfs_2(pointdagmm.model, data, device, args.n_gmm, args.percent, with_card,with_rank, lamda_hat, card_mle)

    torch.cuda.empty_cache()
plt.show()
#plt.savefig('roc_3.svg', format='svg',dpi=300)
# %%
mean_dataset_rk_l2 = torch.tensor(mean_dataset_rk_l2)
mean_dataset_rf = torch.tensor(mean_dataset_rf)
mean_dataset_nv = torch.tensor(mean_dataset_nv)

# mean_rocauc_rk_csd=torch.tensor(mean_dataset_rf)
mean_rocauc_rk_l2 = torch.tensor(mean_dataset_rk_l2)
mean_rocauc_nv = torch.tensor(rocauc_dataset_nv)
mean_rocauc_rf = torch.tensor(rocauc_dataset_rf)

mean_rocauc2_nv = torch.tensor(rocauc2_dataset_nv)
mean_rocauc2_rf = torch.tensor(rocauc2_dataset_rf)
mean_rocauc2_rk_l2 = torch.tensor(rocauc2_dataset_rk_l2)

cardauc_dataset_rk_l2 = torch.tensor(cardauc_dataset_rk_l2)
cardauc_dataset_nv = torch.tensor(cardauc_dataset_nv)


print('====================Mean ROCAUC of mvtec dataset objects================')
print('avg of naive is {:0.2f}'.format(mean_rocauc_nv.mean().numpy()))
print('avg of rf is {:0.2f}'.format(mean_rocauc_rf.mean().numpy()))

print('avg of rank l2 is {:0.2f}'.format(mean_rocauc_rk_l2.float().mean().numpy()))

print('====================Mean ROCAUC of mvtec dataset objects negative mah================')
print('avg of naive is {:0.2f}'.format(mean_rocauc2_nv.mean().numpy()))
print('avg of rf is {:0.2f}'.format(mean_rocauc2_rf.mean().numpy()))
# print('avg of rank csd is {:0.2f}'.format(mean_rocauc2_rk_csd.mean().numpy()))
print('avg of rank l2 is {:0.2f}'.format(mean_rocauc2_rk_l2.mean().numpy()))



