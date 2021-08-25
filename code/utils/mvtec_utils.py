import numpy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from kpdataset.mvteckp import mvtec_data
from kpdataset.mvteck_cvs import mvtec_data_cvs
from torch.distributions import multivariate_normal as mm
import cv2
from torch.utils.data import TensorDataset
import scipy.io as scio

def lf_npy_loader(path,max_card=None):

    sample = np.load(path,allow_pickle=True)['descs']
    if max_card is not None:
        sample=sample
    else:
        if sample.ndim==0:
            sample=torch.from_numpy(np.zeros([1,128],dtype=float))
        else:
            sample=torch.from_numpy(sample)
    return sample
def lf_npy_loader_cvs(path,card_id=None,max_card=None):

    sample = np.load(path,allow_pickle=True)['descs']
    if max_card is not None:
        sample=sample
    else:
        if sample.ndim==0:
            sample=torch.from_numpy(np.zeros([1,128],dtype=float))
        else:
            sample=torch.from_numpy(sample)
            sample=sample[card_id]
    return sample

def sp_npy_loader(path,max_card=None, pts_mean=None):
    pts = np.load(path,allow_pickle=True)['prob']
    if pts_mean is not None:
        pts_mask=torch.tensor([pts[i][2] for i in range(pts.shape[0])])>=pts_mean
    #sample = np.load(path,allow_pickle=True)['desc']
    sample = np.load(path, allow_pickle=True)['norm_desc']

    if pts_mean is not None:
        sample=sample[pts_mask,:]
    # load original image
    if max_card is not None:
        sample=sample
    else:
        if sample.ndim==0:
            sample=torch.from_numpy(np.zeros([1,256],dtype=float))
        else:

            sample=torch.from_numpy(sample)

    return sample

def sp_npy_loader_pts(path,max_card=None, pts_mean=None):
    pts = np.load(path,allow_pickle=True)['pts']
    if pts_mean is not None:
        pts_mask=torch.tensor([pts[i][2] for i in range(pts.shape[0])])>=pts_mean


    return pts
def sp_pts_loader(path,max_card=None):
    pts = np.load(path,allow_pickle=True)['prob']
    return pts


def sp_npy_loader_sift(path,max_card=None):

    sample = np.load(path,allow_pickle=True)['descriptor']
    # load original image
    if max_card is not None:
        sample=sample
    else:
        if sample.ndim==0:
            sample=torch.from_numpy(np.zeros([1,128],dtype=float))
        else:
            sample=torch.from_numpy(sample)/255
    return sample

def r2d2_loader(path,max_card=None):

    sample = np.load(path,allow_pickle=True)['descriptors']
    if max_card is not None:
        sample=sample
    else:
        if sample.ndim==0:
            sample=torch.from_numpy(np.zeros([1,128],dtype=float))
        else:
            sample=torch.from_numpy(sample)
    return sample

def d2_mat_loader(path,max_card=None):
    sample =scio.loadmat(path)
    sample=sample['descriptors']
    if max_card is not None:
        sample=sample
    else:
        if sample.ndim==0:
            print('loading image with no keypoints')
            sample=torch.from_numpy(np.zeros([1,512],dtype=float))
        else:
            sample=torch.from_numpy(sample)
    return sample

def d2_mat_loader_cvs(path,card_id=None,max_card=None):
    sample =scio.loadmat(path)
    sample=sample['descriptors']
    if max_card is not None:
        sample=sample
    else:
        if sample.ndim==0:
            sample=torch.from_numpy(np.zeros([1,512],dtype=float))
        else:
            sample=torch.from_numpy(sample)
            if card_id is not None:
                sample=sample[int(card_id)]
    return sample


def get_cardinality(your_dataloader):
    card_train_data = []
    for i, batch in enumerate(your_dataloader):
        card_train_data.append(batch[2])
    card_train_data
    train_cardinality=card_train_data
    max_card=torch.max(torch.tensor(train_cardinality))
    return train_cardinality,max_card

def get_cardinality_dataset(your_dataloader):
    card_train_data = []
    data_tensor=[]
    data_tensor_label=[]
    for i, batch in enumerate(your_dataloader):
        card_train_data.append(batch[2])
        data_tensor.append(batch[0].squeeze(dim=0))
        data_tensor_label.append(torch.zeros(batch[2]))

    data_tensor = torch.cat(data_tensor)
    data_tensor_label=torch.cat(data_tensor_label)
    card_train_data
    train_cardinality=card_train_data
    max_card=torch.max(torch.tensor(train_cardinality))
    return train_cardinality,max_card,data_tensor,data_tensor_label
def get_pts_mean(your_dataloader):
    pts_prob_mean = []
    for i, batch in enumerate(your_dataloader):
        sample_pts,_,_= batch
        sample_pts=sample_pts.squeeze(dim=0)
        pts_mean = torch.tensor([sample_pts[i][2] for i in range(sample_pts.shape[0])]).mean().unsqueeze(dim=0)
        pts_prob_mean.append(pts_mean)
    pts_prob_mean=torch.tensor(pts_prob_mean)
    pts_mean=pts_prob_mean.mean()
    return pts_mean

def get_max_sp(your_dataloader):
    max_train_sp = []
    min_train_sp = []
    for i, batch in enumerate(your_dataloader):
        sample_desc, _, _ = batch
        max_train_sp.append(sample_desc.max())
        min_train_sp.append(sample_desc.min())

    max_desc=torch.tensor(max_train_sp).max()
    min_desc=torch.tensor(min_train_sp).min()
    return max_desc,min_desc

def compute_CS_div(train_phi,train_mu,train_cov):
    #compute CauchySchwarzDivergence
    #number of components=
    #train_mu.shape=[K,z]
    #train_cov.shape=[K,z,z]
    n_k=train_phi.shape[0]
    C=0 #C= integral(p(x)*p(x)*dx
    if n_k==1:
        mu_k=train_mu
        cov_k=2*train_cov
        from torch.distributions import MultivariateNormal
        from torch.distributions import multivariate_normal

        gmm = multivariate_normal.MultivariateNormal(mu_k, cov_k)
        #gmm=MultivariateNormal(mu_k,cov_k)
        C=torch.exp(gmm.log_prob(mu_k)) # return rqt(integral p(x)p(x)dx)
    else:
        for i in range(n_k):
            mu_k=train_mu[i]
            cov_k = 2*train_cov[i]
            det_cov=torch.det(cov_k)
            gmcd=torch.tensor([2*np.pi]).pow(mu_k.shape[-1]/2).to(device=cov_k.device)*det_cov.sqrt()
            C = C + (train_phi[i].pow(2)) *(1/gmcd)
    return C

def compute_z_l2_norm(mu_i,mu_j,cov_i,cov_j):

    #if torch.mean(mu_i-mu_j).cpu().numpy() == 0:
    if torch.equal(mu_i,mu_j):
        exp_term = 0
        zij = torch.det(2 * np.pi * (cov_i + cov_j)).pow(-0.5)
    else:
        sub=mu_i-mu_j
        exp_term = -0.5*sub.t()*torch.inverse(cov_i+cov_j)*(mu_i-mu_j)
        zij = torch.det(2 * np.pi * (cov_i + cov_j)).pow(-0.5) * torch.exp(exp_term)

    return zij


def compute_z_l2_norm2(mu_i,mu_j,cov_i,cov_j):

    #if torch.mean(mu_i-mu_j).cpu().numpy() == 0:
    if torch.equal(mu_i,mu_j):
        exp_term = 0
        zij = torch.det(2 * np.pi * (cov_i + cov_j)).pow(-0.5)
    else:
        sub=mu_i-mu_j
        exp_term = -0.5*torch.sum(torch.sum(sub.unsqueeze(dim=-1)*torch.inverse(cov_i+cov_j),dim=-1)*sub,dim=-1)
        zij = torch.det(2 * np.pi * (cov_i + cov_j)).pow(-0.5) * torch.exp(exp_term)

    return zij



def compute_l2_norm(train_phi,train_mu,train_cov):
    #compute l2 nrom of Gaussian mixture.or single Gaussian Please, see my derivation of l2_norm by ammar kamooona
    #number of components=
    #train_mu.shape=[K,z]
    #train_cov.shape=[K,z,z]
    n_k=train_phi.shape[0]
    if n_k==1:
        C=torch.det(2 * np.pi * (2*train_cov)).pow(-0.5) #look to my derivation of single gaussian l2-norm

    else:

        C=0 #C= integral(p(x)*p(x)*dx)
        for i in range(n_k):
            mu_i=train_mu[i]
            cov_i = train_cov[i]
            for j in range(n_k):
                if j==i :
                    continue
                else:
                    mu_j=train_mu[j]
                    cov_j=train_cov[j]
                    phi_j=train_phi[j]
                C=C+train_phi[i]*train_phi[j]*compute_z_l2_norm2(mu_i,mu_j,cov_i,cov_j)
            C=C+train_phi[i]*train_phi[i]*compute_z_l2_norm2(mu_i,mu_i,cov_i,cov_i)

    return C # Return the l2 norm integral p(x)*p(x)dx


def get_mvtec_data(args):
    """Returning train and test dataloaders."""
    data_dir = args.data_dir
    feat_type=args.feat_type
    #load the keypoints for first time to get the max lengths
    if feat_type=='lf_net':
        feat_loader=lf_npy_loader
        ext='.npz'
    if feat_type=='d2_net':
        feat_loader = d2_mat_loader
        ext = '.mat'
    if feat_type=='r2d2':
        feat_loader = r2d2_loader
        ext = '.r2d2'
    if feat_type=='sp':
        feat_loader=sp_npy_loader
        ext='.npz'
    if feat_type=='sp-sift':
        feat_loader=sp_npy_loader_sift
        ext='.npz'

    # the following option only for superpoint
    if feat_type=='sp':

        train_set = mvtec_data(root=data_dir, loader=sp_pts_loader,args=args,extensions=ext, classes=['train'], only_card=False)
        trainloader_set = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False)
        mean_pts=get_pts_mean(trainloader_set)

        #filter_out points with prob less than mean_pts
        if args.pick_top_k==1:# FIND OUT the pts mean prob of traning samples
            train_set = mvtec_data(root=data_dir, loader=sp_npy_loader, extensions=ext, classes=['train'], only_card=False,filter_pts=mean_pts)
        else:
            train_set = mvtec_data(root=data_dir, loader=sp_npy_loader, extensions=ext, classes=['train'],
                                   only_card=False)
        trainloader_set = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False)
        max_sample, min_sample = get_max_sp(trainloader_set)
        train_cardinality, max_card_train = get_cardinality(trainloader_set)
        # load the padding training dataset
        if args.pick_top_k == 1:  #
            train = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['train'], only_card=False,max_card=max_card_train,filter_pts=mean_pts,sample_max=max_sample,sample_min=min_sample)
        else:
            train = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['train'], only_card=False,
                               max_card=max_card_train, sample_max=max_sample,
                               sample_min=min_sample)
        # the padded train dataloader

        if args.batch_size == 'full':
            args.batch_size = len(trainloader_set)
            dataloader_train = DataLoader(train, batch_size=len(trainloader_set), shuffle=True, num_workers=0)
        else:

            dataloader_train = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)

        # the test with padding, the test set with no padding
        if args.pick_top_k == 1:
            test_set = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['test'], only_card=False,filter_pts=mean_pts,sample_max=max_sample,sample_min=min_sample)
        else:
            test_set = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['test'], only_card=False, sample_max=max_sample, sample_min=min_sample)
        testloader_set = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
        # the padded test set using max_test
        if args.pick_top_k == 1:
            test = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['test'], only_card=False,filter_pts=mean_pts,sample_max=max_sample,sample_min=min_sample)
        else:
            test = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['test'], only_card=False, sample_max=max_sample, sample_min=min_sample)
        dataloader_test = DataLoader(test, batch_size=1, shuffle=False, num_workers=0)
        if args.pick_top_k == 1:
            train_set_2 = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['train'], only_card=True,filter_pts=mean_pts,sample_max=max_sample,sample_min=min_sample)
        else:
            train_set_2 = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['train'],
                                     only_card=True, sample_max=max_sample, sample_min=min_sample)

        trainloader_set = torch.utils.data.DataLoader(train_set_2, batch_size=len(train_set), shuffle=False)

        # this dataloader will be used for evaluation
        trainloader_eval = torch.utils.data.DataLoader(train, batch_size=1, shuffle=False)
        # trainloader_set = torch.utils.data.DataLoader(train_set_2, batch_size=1, shuffle=False)

        # return dataloader_train, dataloader_test, trainloader_set, testloader_set, trainloader_eval

        return dataloader_train, testloader_set, trainloader_set, testloader_set, trainloader_eval

    # the below dataset class has different number of cardinallity
    train_set=mvtec_data(root=data_dir,loader=feat_loader,extensions=ext,classes=['train'],only_card=False)
    ##dataloader using batch_size=1 to get the max cardinlaity for purpose of padding
    trainloader_set = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False)
    #get the max lenght of features
    train_cardinality, max_card_train = get_cardinality(trainloader_set)
    #load the padding training dataset using max length(max_card_train)
    train = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['train'], only_card=False,max_card=max_card_train)

    # splite the traning dataset into validation and testing
    #train_set, val_set = torch.utils.data.random_split(train, [50000, 10000])
    #the padded train dataloader


    if args.batch_size=='full':
        args.batch_size=len(trainloader_set)
        dataloader_train = DataLoader(train, batch_size=len(trainloader_set), shuffle=True, num_workers=0)
    elif args.batch_size == 'half':
        args.batch_size = len(trainloader_set)//2
        dataloader_train = DataLoader(train, batch_size=(len(trainloader_set)//2), shuffle=True, num_workers=0)

    else:
        dataloader_train = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)


    # the test with padding, the test set with no padding
    test_set = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['test'], only_card=False)
    testloader_set = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    # the padded test set using max_test
    test = mvtec_data(root=data_dir,loader=feat_loader,extensions=ext,classes=['test'],only_card=False)
    dataloader_test = DataLoader(test, batch_size=1, shuffle=False, num_workers=0)
    train_set_2 = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['train'], only_card=True)
    cardinality_loader = torch.utils.data.DataLoader(train_set_2, batch_size=len(train_set), shuffle=False)

    # this dataloader will be used for evaluation
    trainloader_eval = torch.utils.data.DataLoader(train, batch_size=1, shuffle=False)
    #trainloader_set = torch.utils.data.DataLoader(train_set_2, batch_size=1, shuffle=False)

    #return dataloader_train, dataloader_test, trainloader_set, testloader_set, trainloader_eval
    print('retutn train_loader, teste_loader, train_loader_batch=1, cardinality loader, train_loader_eval')

    return dataloader_train, testloader_set, trainloader_set, cardinality_loader, trainloader_eval

# this version map test data into max
def get_mvtec_data_2(args):
    """Returning train and test dataloaders."""
    data_dir=args.data_dir
    feat_type=args.feat_type
    #load the keypoints for first time to get the max lengths
    if feat_type=='lf_net':
        feat_loader=lf_npy_loader
        ext='.npz'
    if feat_type=='d2_net' or feat_type=='d2_net4' :
        feat_loader = d2_mat_loader
        ext = '.mat'
    if feat_type=='r2d2':
        feat_loader = r2d2_loader
        ext = '.r2d2'
    if feat_type=='sp':
        feat_loader=sp_npy_loader
        ext='.npz'
    if feat_type=='sp-sift':
        feat_loader=sp_npy_loader_sift
        ext='.npz'

    # the following option only for superpoint
    if feat_type=='sp':

        train_set = mvtec_data(root=data_dir, loader=sp_pts_loader, extensions=ext, classes=['train'], only_card=False)
        trainloader_set = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False)
        mean_pts=get_pts_mean(trainloader_set)

        #filter_out points with prob less than mean_pts
        if args.pick_top_k==1:# FIND OUT the pts mean prob of traning samples
            train_set = mvtec_data(root=data_dir, loader=sp_npy_loader, extensions=ext, classes=['train'], only_card=False,filter_pts=mean_pts)
        else:
            train_set = mvtec_data(root=data_dir, loader=sp_npy_loader, extensions=ext, classes=['train'],
                                   only_card=False)
        trainloader_set = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False)
        max_sample, min_sample = get_max_sp(trainloader_set)
        train_cardinality, max_card_train = get_cardinality(trainloader_set)
        # load the padding training dataset
        if args.pick_top_k == 1:  #
            train = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['train'], only_card=False,max_card=max_card_train,filter_pts=mean_pts,sample_max=max_sample,sample_min=min_sample)
        else:
            train = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['train'], only_card=False,
                               max_card=max_card_train, sample_max=max_sample,
                               sample_min=min_sample)
        # the padded train dataloader

        if args.batch_size == 'full':
            args.batch_size = len(trainloader_set)
            dataloader_train = DataLoader(train, batch_size=len(trainloader_set), shuffle=True, num_workers=0)
        else:

            dataloader_train = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)

        # the test with padding, the test set with no padding
        if args.pick_top_k == 1:
            test_set = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['test'], only_card=False,filter_pts=mean_pts,sample_max=max_sample,sample_min=min_sample)
        else:
            test_set = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['test'], only_card=False, sample_max=max_sample, sample_min=min_sample)
        testloader_set = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
        # the padded test set using max_test
        if args.pick_top_k == 1:
            test = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['test'], only_card=False,filter_pts=mean_pts,sample_max=max_sample,sample_min=min_sample)
        else:
            test = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['test'], only_card=False, sample_max=max_sample, sample_min=min_sample)
        dataloader_test = DataLoader(test, batch_size=1, shuffle=False, num_workers=0)
        if args.pick_top_k == 1:
            train_set_2 = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['train'], only_card=True,filter_pts=mean_pts,sample_max=max_sample,sample_min=min_sample)
        else:
            train_set_2 = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['train'],
                                     only_card=True, sample_max=max_sample, sample_min=min_sample)

        trainloader_set = torch.utils.data.DataLoader(train_set_2, batch_size=len(train_set), shuffle=False)

        # this dataloader will be used for evaluation
        trainloader_eval = torch.utils.data.DataLoader(train, batch_size=1, shuffle=False)
        # trainloader_set = torch.utils.data.DataLoader(train_set_2, batch_size=1, shuffle=False)

        # return dataloader_train, dataloader_test, trainloader_set, testloader_set, trainloader_eval

        return dataloader_train, testloader_set, trainloader_set, testloader_set, trainloader_eval


    train_set=mvtec_data(root=data_dir,loader=feat_loader,extensions=ext,classes=['train'],only_card=False)
    ##dataloader using batch_size=1 to get the max cardinlaity for purpose of padding
    trainloader_set = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False)
    #get the max lenght of features
    train_cardinality, max_card_train = get_cardinality(trainloader_set)
    #load the padding training dataset
    train = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['train'], only_card=False,max_card=max_card_train)
    #the padded train dataloader

    if args.batch_size=='full':
        args.batch_size=len(trainloader_set)
        dataloader_train = DataLoader(train, batch_size=len(trainloader_set), shuffle=True, num_workers=0)
    elif args.batch_size == 'half':
        args.batch_size = len(trainloader_set)//2
        dataloader_train = DataLoader(train, batch_size=(len(trainloader_set)//2), shuffle=True, num_workers=0)

    else:

        dataloader_train = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)


    # the test with padding, the test set with no padding
    test_set = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['test'], only_card=False)
    testloader_set = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    # the padded test set using max_test
    test = mvtec_data(root=data_dir,loader=feat_loader,extensions=ext,classes=['test'],only_card=False)
    dataloader_test = DataLoader(test, batch_size=1, shuffle=False, num_workers=0)
    train_set_2 = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['train'], only_card=True)
    trainloader_set = torch.utils.data.DataLoader(train_set_2, batch_size=len(train_set), shuffle=False)

    # this dataloader will be used for evaluation
    trainloader_eval = torch.utils.data.DataLoader(train, batch_size=1, shuffle=False)
    #trainloader_set = torch.utils.data.DataLoader(train_set_2, batch_size=1, shuffle=False)

    #return dataloader_train, dataloader_test, trainloader_set, testloader_set, trainloader_eval

    return dataloader_train, testloader_set, trainloader_set, testloader_set, trainloader_eval
def get_mvtec_data_sp(args):
    """extract kp"""
    #data_dir=args.data_im
    #feat_type=args.feat_type
    data_dir = args.data_dir
    feat_type = args.feat_type
    #load the keypoints for first time to get the max lengths
    if feat_type=='lf_net':
        feat_loader=lf_npy_loader
        ext='.npz'
    if feat_type=='d2_net':
        feat_loader = d2_mat_loader
        ext = '.mat'
    if feat_type=='r2d2':
        feat_loader = r2d2_loader
        ext = '.r2d2'
    if feat_type=='sp':
        feat_loader=sp_npy_loader
        ext='.npz'
    if feat_type=='sp-sift':
        feat_loader=sp_npy_loader_pts
        ext='.npz'

    train_set=mvtec_data(root=data_dir,loader=feat_loader,extensions=ext,classes=['train'],only_card=False)
    ##dataloader using batch_size=1 to get the max cardinlaity for purpose of padding
    trainloader_set = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False)
    #get the max lenght of features
    train_cardinality, max_card_train = get_cardinality(trainloader_set)
    #load the padding training dataset
    train = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['train'], only_card=False,max_card=max_card_train)
    #the padded train dataloader

    if args.batch_size=='full':
        args.batch_size=len(trainloader_set)
        dataloader_train = DataLoader(train, batch_size=len(trainloader_set), shuffle=True, num_workers=0)
    else:

        dataloader_train = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)


    # the test with padding, the test set with no padding
    test_set = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['test'], only_card=False)
    testloader_set = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    # the padded test set using max_test
    test = mvtec_data(root=data_dir,loader=feat_loader,extensions=ext,classes=['test'],only_card=False)
    dataloader_test = DataLoader(test, batch_size=1, shuffle=False, num_workers=0)
    train_set_2 = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['train'], only_card=True)
    trainloader_set = torch.utils.data.DataLoader(train_set_2, batch_size=len(train_set), shuffle=False)

    # this dataloader will be used for evaluation
    trainloader_eval = torch.utils.data.DataLoader(train, batch_size=1, shuffle=False)
    #trainloader_set = torch.utils.data.DataLoader(train_set_2, batch_size=1, shuffle=False)

    #return dataloader_train, dataloader_test, trainloader_set, testloader_set, trainloader_eval

    return dataloader_train, testloader_set, trainloader_set, testloader_set, trainloader_eval

def get_mvtec_data_cvs(args):
    """Returning train and test dataloaders  using cvs file."""

    if args.use_cvs_file==1:
        data_dir_cvs=args.data_dir_cvs+args.object+'/'

    data_dir = args.data_dir
    feat_type=args.feat_type
    #load the keypoints for first time to get the max lengths
    if feat_type=='lf_net':
        feat_loader_cvs=lf_npy_loader_cvs
        feat_loader=lf_npy_loader
        ext='.npz'
    if feat_type=='d2_net':
        feat_loader_cvs = d2_mat_loader_cvs
        feat_loader = d2_mat_loader
        ext = '.mat'
    if feat_type=='r2d2':
        feat_loader_cvs = r2d2_loader
        feat_loader = r2d2_loader
        ext = '.r2d2'
    if feat_type=='sp':
        feat_loader=sp_npy_loader
        ext='.npz'
    if feat_type=='sp-sift':
        feat_loader=sp_npy_loader_sift
        ext='.npz'

    # the following option only for superpoint
    if feat_type=='sp':

        train_set = mvtec_data_cvs(root=data_dir, loader=sp_pts_loader,args=args,extensions=ext, classes=['train'], only_card=False)
        trainloader_set = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False)
        mean_pts=get_pts_mean(trainloader_set)

        #filter_out points with prob less than mean_pts
        if args.pick_top_k==1:# FIND OUT the pts mean prob of traning samples
            train_set = mvtec_data(root=data_dir, loader=sp_npy_loader, extensions=ext, classes=['train'], only_card=False,filter_pts=mean_pts)
        else:
            train_set = mvtec_data(root=data_dir, loader=sp_npy_loader, extensions=ext, classes=['train'],
                                   only_card=False)
        trainloader_set = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False)
        max_sample, min_sample = get_max_sp(trainloader_set)
        train_cardinality, max_card_train = get_cardinality(trainloader_set)
        # load the padding training dataset
        if args.pick_top_k == 1:  #
            train = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['train'], only_card=False,max_card=max_card_train,filter_pts=mean_pts,sample_max=max_sample,sample_min=min_sample)
        else:
            train = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['train'], only_card=False,
                               max_card=max_card_train, sample_max=max_sample,
                               sample_min=min_sample)
        # the padded train dataloader

        if args.batch_size == 'full':
            args.batch_size = len(trainloader_set)
            dataloader_train = DataLoader(train, batch_size=len(trainloader_set), shuffle=True, num_workers=0)
        else:

            dataloader_train = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)

        # the test with padding, the test set with no padding
        if args.pick_top_k == 1:
            test_set = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['test'], only_card=False,filter_pts=mean_pts,sample_max=max_sample,sample_min=min_sample)
        else:
            test_set = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['test'], only_card=False, sample_max=max_sample, sample_min=min_sample)
        testloader_set = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
        # the padded test set using max_test
        if args.pick_top_k == 1:
            test = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['test'], only_card=False,filter_pts=mean_pts,sample_max=max_sample,sample_min=min_sample)
        else:
            test = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['test'], only_card=False, sample_max=max_sample, sample_min=min_sample)
        dataloader_test = DataLoader(test, batch_size=1, shuffle=False, num_workers=0)
        if args.pick_top_k == 1:
            train_set_2 = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['train'], only_card=True,filter_pts=mean_pts,sample_max=max_sample,sample_min=min_sample)
        else:
            train_set_2 = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['train'],
                                     only_card=True, sample_max=max_sample, sample_min=min_sample)

        trainloader_set = torch.utils.data.DataLoader(train_set_2, batch_size=len(train_set), shuffle=False)

        # this dataloader will be used for evaluation
        trainloader_eval = torch.utils.data.DataLoader(train, batch_size=1, shuffle=False)
        # trainloader_set = torch.utils.data.DataLoader(train_set_2, batch_size=1, shuffle=False)

        # return dataloader_train, dataloader_test, trainloader_set, testloader_set, trainloader_eval

        return dataloader_train, testloader_set, trainloader_set, testloader_set, trainloader_eval


    # build cvs dataset loader usinf cvs file
    train=mvtec_data_cvs(root=data_dir_cvs,loader=feat_loader_cvs,args=args,extensions=ext,classes=['train'],only_card=False)
    if args.batch_size == 'full':
        dataloader_train = DataLoader(train, batch_size=len(train), shuffle=True, num_workers=8)
    elif args.batch_size == 'half':
        dataloader_train = DataLoader(train, batch_size=len(train) // 2, shuffle=True, num_workers=8)
    else:
        dataloader_train = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    #train_set for evaluation purpose return set of keypoint descriptors for each image
    ##train_eval = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['train'], only_card=False)
    # this dataloader will be used for inference and return a set of keypoints
    ###trainloader_eval = torch.utils.data.DataLoader(train_eval, batch_size=1, shuffle=False)

    # train_card dataset  and loader
    ##card_train= mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['train'], only_card=True)
    ###card_trainloader= DataLoader(card_train, batch_size=len(card_train), shuffle=False)
    # the test dataset with set of keypoints
    ##test_set = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['test'], only_card=False)
    ##testloader_set = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)


    #return dataloader_train, testloader_set, card_trainloader,train, trainloader_eval
    return dataloader_train

# this version map test data into max
def get_mvtec(args,train_only=True,with_vaild=True):
    """Returning train and test dataloaders."""
    data_dir = args.data_dir
    feat_type=args.feat_type
    #load the keypoints for first time to get the max lengths
    if feat_type=='lf_net':
        feat_loader=lf_npy_loader
        ext='.npz'
    if feat_type=='d2_net' or feat_type =='d2_net4':
        feat_loader = d2_mat_loader
        ext = '.mat'
    if feat_type=='r2d2':
        feat_loader = r2d2_loader
        ext = '.r2d2'
    if feat_type=='sp':
        feat_loader=sp_npy_loader
        ext='.npz'
    if feat_type=='sp-sift':
        feat_loader=sp_npy_loader_sift
        ext='.npz'

    # the following option only for superpoint
    if feat_type=='sp':

        train_set = mvtec_data(root=data_dir, loader=sp_pts_loader,args=args,extensions=ext, classes=['train'], only_card=False)
        trainloader_set = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False)
        mean_pts=get_pts_mean(trainloader_set)

        #filter_out points with prob less than mean_pts
        if args.pick_top_k==1:# FIND OUT the pts mean prob of traning samples
            train_set = mvtec_data(root=data_dir, loader=sp_npy_loader, extensions=ext, classes=['train'], only_card=False,filter_pts=mean_pts)
        else:
            train_set = mvtec_data(root=data_dir, loader=sp_npy_loader, extensions=ext, classes=['train'],
                                   only_card=False)
        trainloader_set = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False)
        max_sample, min_sample = get_max_sp(trainloader_set)
        train_cardinality, max_card_train = get_cardinality(trainloader_set)
        # load the padding training dataset
        if args.pick_top_k == 1:  #
            train = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['train'], only_card=False,max_card=max_card_train,filter_pts=mean_pts,sample_max=max_sample,sample_min=min_sample)
        else:
            train = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['train'], only_card=False,
                               max_card=max_card_train, sample_max=max_sample,
                               sample_min=min_sample)
        # the padded train dataloader

        if args.batch_size == 'full':
            args.batch_size = len(trainloader_set)
            dataloader_train = DataLoader(train, batch_size=len(trainloader_set), shuffle=True, num_workers=0)
        else:

            dataloader_train = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)

        # the test with padding, the test set with no padding
        if args.pick_top_k == 1:
            test_set = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['test'], only_card=False,filter_pts=mean_pts,sample_max=max_sample,sample_min=min_sample)
        else:
            test_set = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['test'], only_card=False, sample_max=max_sample, sample_min=min_sample)
        testloader_set = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
        # the padded test set using max_test
        if args.pick_top_k == 1:
            test = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['test'], only_card=False,filter_pts=mean_pts,sample_max=max_sample,sample_min=min_sample)
        else:
            test = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['test'], only_card=False, sample_max=max_sample, sample_min=min_sample)
        dataloader_test = DataLoader(test, batch_size=1, shuffle=False, num_workers=0)
        if args.pick_top_k == 1:
            train_set_2 = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['train'], only_card=True,filter_pts=mean_pts,sample_max=max_sample,sample_min=min_sample)
        else:
            train_set_2 = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['train'],
                                     only_card=True, sample_max=max_sample, sample_min=min_sample)

        trainloader_set = torch.utils.data.DataLoader(train_set_2, batch_size=len(train_set), shuffle=False)

        # this dataloader will be used for evaluation
        trainloader_eval = torch.utils.data.DataLoader(train, batch_size=1, shuffle=False)
        # trainloader_set = torch.utils.data.DataLoader(train_set_2, batch_size=1, shuffle=False)

        # return dataloader_train, dataloader_test, trainloader_set, testloader_set, trainloader_eval

        return dataloader_train, testloader_set, trainloader_set, testloader_set, trainloader_eval


    # try to load dataset using batch_size one to find the max length
    train_set=mvtec_data(root=data_dir,loader=feat_loader,extensions=ext,classes=['train'],only_card=False)
    ##dataloader using batch_size=1 to get the max cardinlaity for purpose of padding
    trainloader_set = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False)
    #get the max lenght of features
    train_cardinality, max_card_train = get_cardinality(trainloader_set)
    train_cardinality, max_card_train, dataset_matrix,dataset_matrix_label = get_cardinality_dataset(trainloader_set)
    print('the totat number of training keypoints=', torch.cat(train_cardinality).sum(dim=0).cpu().numpy())
    #my_dataset = TensorDataset(dataset_matrix,dataset_matrix_label,train_cardinality)
    # obtain the max samples and min samples to normalise
    max_sample, min_sample = get_max_sp(trainloader_set)
    #load the padding training dataset
    if args.normalise_data==1:

        train = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['train'], only_card=False,max_card=max_card_train,sample_max=max_sample,
                               sample_min=min_sample)
        #use this for evaluation of the set likelihood
        train_set_eval=mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['train'], only_card=False,sample_max=max_sample,
                               sample_min=min_sample)

    else:
        train = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['train'], only_card=False,
                       max_card=max_card_train)
        # use this for evaluation of the set likelihood
        train_set_eval = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['train'], only_card=False)


    # splite the traning dataset into validation and testing using 20% pecent of normal samples
    if with_vaild:
        train_set, val_set = torch.utils.data.random_split(train, [int(len(train)-(int(0.2*len(train)))), int(0.2*len(train))])
    #the padded train dataloader
    else:
        train_set=train

    if args.batch_size=='full':
        args.batch_size=len(trainloader_set)
        dataloader_train = DataLoader(train_set, batch_size=len(trainloader_set), shuffle=True, num_workers=0)
        if with_vaild:
            dataloader_vald = DataLoader(val_set, batch_size=len(trainloader_set), shuffle=True, num_workers=0)
    elif args.batch_size == 'half':
        args.batch_size = len(trainloader_set)//2
        dataloader_train = DataLoader(train_set, batch_size=(len(trainloader_set)//2), shuffle=True, num_workers=0)
        if with_vaild:
            dataloader_vald = DataLoader(val_set,batch_size=(len(trainloader_set)//2), shuffle=True, num_workers=0)

    else:
        dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
        if with_vaild:
            dataloader_vald = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=0)




    # the test with padding, the test set with no padding
    if args.normalise_data == 1:
        test_set = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['test'], only_card=False,sample_max=max_sample,
                               sample_min=min_sample)
    else:
        test_set = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['test'], only_card=False)

    testloader_set = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    # the padded test set using max_test
    if args.normalise_data==1:
        test = mvtec_data(root=data_dir,loader=feat_loader,extensions=ext,classes=['test'],only_card=False,sample_max=max_sample,
                               sample_min=min_sample)
    else:

        test = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['test'], only_card=False)
    dataloader_test = DataLoader(test, batch_size=1, shuffle=False, num_workers=0)

    train_set_2 = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['train'], only_card=True)
    card_trainer = torch.utils.data.DataLoader(train_set_2, batch_size=len(train_set_2), shuffle=False)

    # this dataloader will be used for evaluation
    trainloader_eval = torch.utils.data.DataLoader(train_set_eval, batch_size=1, shuffle=False)
    #trainloader_set = torch.utils.data.DataLoader(train_set_2, batch_size=1, shuffle=False)

    #return dataloader_train, dataloader_test, trainloader_set, testloader_set, trainloader_eval
    if train_only:
        print('return train_loader, vailidation loader, card loader')
        if with_vaild:
            return dataloader_train, dataloader_vald, card_trainer
        return dataloader_train, card_trainer

    return dataloader_train, testloader_set, card_trainer, testloader_set, trainloader_eval


def get_mvtec_v2(args,train_only=True,with_vaild=True):
    """Returning train and test dataloaders. This version ensure each batch carry keypoints rather than images"""
    data_dir = args.data_dir
    feat_type=args.feat_type
    #load the keypoints for first time to get the max lengths
    if feat_type=='lf_net':
        feat_loader=lf_npy_loader
        ext='.npz'
    if feat_type=='d2_net' or feat_type=='d2_net2' or feat_type=='d2_net4' or feat_type=='d2_net6':
        feat_loader = d2_mat_loader
        ext = '.mat'
    if feat_type=='r2d2':
        feat_loader = r2d2_loader
        ext = '.r2d2'
    if feat_type=='sp':
        feat_loader=sp_npy_loader
        ext='.npz'
    if feat_type=='sp-sift':
        feat_loader=sp_npy_loader_sift
        ext='.npz'




    # load image keypoint dataset using batch_size one to find the max length (return keypoints of each image)
    train_img_kp=mvtec_data(root=data_dir,loader=feat_loader,extensions=ext,classes=['train'],only_card=False)
    ##dataloader using batch_size=1 to get the max cardinlaity for purpose of padding
    trainloader_set = torch.utils.data.DataLoader(train_img_kp, batch_size=1, shuffle=False)
    # obtain the max samples and min samples to normalise
    max_sample, min_sample = get_max_sp(trainloader_set)
    #lload image normalized or not normalized image keypoit dataset, then load keypoint dataset class
    if args.normalise_data==1:

        train_img_kp = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['train'], only_card=False,sample_max=max_sample,
                               sample_min=min_sample)
        trainloader_set = torch.utils.data.DataLoader(train_img_kp, batch_size=1, shuffle=False)
        # get the max lenght of features, keppoint datasetmatrix
        train_cardinality, max_card_train, dataset_matrix, dataset_matrix_label = get_cardinality_dataset(trainloader_set)
        #convert keypoints datasetmatrix into keppoint datset classs


    else:
        train_img_kp = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['train'], only_card=False)
        trainloader_set = torch.utils.data.DataLoader(train_img_kp, batch_size=1, shuffle=False)
        train_cardinality, max_card_train, dataset_matrix, dataset_matrix_label = get_cardinality_dataset(trainloader_set)
        # convert keypoints datasetmatrix into keppoint datset classs

    my_train_kp_dataset = TensorDataset(dataset_matrix, dataset_matrix_label)

    # splite the traning dataset into validation and testing using 20% pecent of normal samples
    if with_vaild:
        train_set, val_set = torch.utils.data.random_split(my_train_kp_dataset, [int(len(my_train_kp_dataset)-(int(0.2*len(my_train_kp_dataset)))), int(0.2*len(my_train_kp_dataset))])
    #the padded train dataloader
    else:
        train_set=my_train_kp_dataset

    if args.batch_size=='full':
        dataloader_train = DataLoader(train_set, batch_size=len(train_set), shuffle=True, num_workers=0)
        if with_vaild:
            dataloader_vald = DataLoader(val_set, batch_size=len(val_set), shuffle=True, num_workers=0)
    elif args.batch_size == 'half':
        dataloader_train = DataLoader(train_set, batch_size=(len(train_set)//2), shuffle=True, num_workers=0)
        if with_vaild:
            dataloader_vald = DataLoader(val_set,batch_size=(len(val_set)//2), shuffle=True, num_workers=0)

    else:
        dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
        if with_vaild:
            dataloader_vald = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=0)


    # the test with padding, the test set with no padding
    if args.normalise_data == 1:
        test_set = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['test'], only_card=False,sample_max=max_sample,
                               sample_min=min_sample)
    else:
        test_set = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['test'], only_card=False)

    testloader_set = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    # the padded test set using max_test

    train_card = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['train'], only_card=True)
    card_trainer = torch.utils.data.DataLoader(train_card, batch_size=len(train_img_kp), shuffle=False)

    # this dataloader will be used for evaluation
    trainloader_eval = torch.utils.data.DataLoader(train_img_kp, batch_size=1, shuffle=False)
    #trainloader_set = torch.utils.data.DataLoader(train_set_2, batch_size=1, shuffle=False)

    #return dataloader_train, dataloader_test, trainloader_set, testloader_set, trainloader_eval
    if train_only:

        if with_vaild:
            print('return train_loader, vailidation loader, card loader')
            return dataloader_train, dataloader_vald, card_trainer
        print('return train_loader, card loader')
        return dataloader_train, card_trainer
    print('Return train_loader,test_loader batch=1,cardinality loader,train_loader(batch=1)')

    return dataloader_train, testloader_set, card_trainer, testloader_set, trainloader_eval

def get_mvtec_datamatrix(args,train_only=True,with_vaild=True):
    """Returning train datamatrix and test datamatrix"""
    data_dir = args.data_dir
    feat_type=args.feat_type
    #load the keypoints for first time to get the max lengths
    if feat_type=='lf_net':
        feat_loader=lf_npy_loader
        ext='.npz'
    if feat_type=='d2_net' or feat_type=='d2_net4':
        feat_loader = d2_mat_loader
        ext = '.mat'
    if feat_type=='r2d2':
        feat_loader = r2d2_loader
        ext = '.r2d2'
    if feat_type=='sp':
        feat_loader=sp_npy_loader
        ext='.npz'
    if feat_type=='sp-sift':
        feat_loader=sp_npy_loader_sift
        ext='.npz'

    # load image keypoint dataset using batch_size one to find the max length (return keypoints of each image)
    train_img_kp=mvtec_data(root=data_dir,loader=feat_loader,extensions=ext,classes=['train'],only_card=False)
    ##dataloader using batch_size=1 to get the max cardinlaity for purpose of padding
    trainloader_set = torch.utils.data.DataLoader(train_img_kp, batch_size=1, shuffle=False)
    # obtain the max samples and min samples to normalise
    max_sample, min_sample = get_max_sp(trainloader_set)
    #lload image normalized or not normalized image keypoit dataset, then load keypoint dataset class
    if args.normalise_data==1:

        train_img_kp = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['train'], only_card=False,sample_max=max_sample,
                               sample_min=min_sample)
        trainloader_set = torch.utils.data.DataLoader(train_img_kp, batch_size=1, shuffle=False)
        # get the max lenght of features, keppoint datasetmatrix
        train_cardinality, max_card_train, dataset_matrix, dataset_matrix_label = get_cardinality_dataset(trainloader_set)
        #convert keypoints datasetmatrix into keppoint datset classs


    else:
        train_img_kp = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['train'], only_card=False)
        trainloader_set = torch.utils.data.DataLoader(train_img_kp, batch_size=1, shuffle=False)
        train_cardinality, max_card_train, dataset_matrix, dataset_matrix_label = get_cardinality_dataset(trainloader_set)
        # convert keypoints datasetmatrix into keppoint datset classs

    my_train_kp_dataset = TensorDataset(dataset_matrix, dataset_matrix_label)

    # splite the traning dataset into validation and testing using 20% pecent of normal samples
    if with_vaild:
        train_set, val_set = torch.utils.data.random_split(my_train_kp_dataset, [int(len(my_train_kp_dataset)-(int(0.2*len(my_train_kp_dataset)))), int(0.2*len(my_train_kp_dataset))])
    #the padded train dataloader
    else:
        train_set=my_train_kp_dataset



    # the test with padding, the test set with no padding
    if args.normalise_data == 1:
        test_set = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['test'], only_card=False,sample_max=max_sample,
                               sample_min=min_sample)
    else:
        test_set = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['test'], only_card=False)

    testloader_set = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    # the padded test set using max_test

    train_card = mvtec_data(root=data_dir, loader=feat_loader, extensions=ext, classes=['train'], only_card=True)
    card_trainer = torch.utils.data.DataLoader(train_card, batch_size=len(train_img_kp), shuffle=False)

    # this dataloader will be used for evaluation
    trainloader_eval = torch.utils.data.DataLoader(train_img_kp, batch_size=1, shuffle=False)
    #trainloader_set = torch.utils.data.DataLoader(train_set_2, batch_size=1, shuffle=False)

    #return dataloader_train, dataloader_test, trainloader_set, testloader_set, trainloader_eval
    if train_only:
        return dataset_matrix, card_trainer

    return dataset_matrix, testloader_set, card_trainer, testloader_set, trainloader_eval