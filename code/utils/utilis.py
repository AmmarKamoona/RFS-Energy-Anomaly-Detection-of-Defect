import torch
import cv2

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != 'Conv':
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.normal_(m.bias.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.normal_(m.bias.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)

def check_model_grad(model):
    for name, param in model.state_dict().items():
        print(name)
        print(param.requires_grad)
        print(param.weight.grad)

def check_loss_setting():
    input = torch.randn(3, requires_grad=True)
    target = torch.empty(3).random_(2)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(input, target)
    print('print the loss grad true')
    print(loss.requires_grad)#True
    print('printing the loss grad_fn')
    loss.grad_fn#<BinaryCrossEntropyWithLogitsBackward at 0x7f5ec005bd68>
    print('print if the loss is a not leaf')
    print(loss.is_leaf)#False
    loss.backward()


#this will be used for all evaluation and
def eval_MHD_like(model, dataloaders, device, args, with_card, with_rank, lamda_hat, card_mle,card_std):
    import numpy as np
    """Testing the Mahal distance"""
    train_loader_b1, dataloader_test,card_test_list,test_labels= dataloaders

    # dataloader_train, testloader_set, card_trainer, testloader_set, trainloader_eval
    train_mu, train_cov = model
    train_mu = torch.tensor(train_mu, dtype=float).to(device)
    train_cov = torch.tensor(train_cov, dtype=float).to(device)
    print('Testing...')
    pecent = args.percent

    # Obtaining Labels and energy scores for train data
    energy_train = []
    labels_train = []
    card_energy_train = []
    dist_train = []
    dist_train_v2 = []
    dist_train_log = []
    dist_train_log_sq = []

    def compute_CS_div(train_phi, train_mu, train_cov):
        # compute CauchySchwarzDivergence
        # number of components=
        # train_mu.shape=[K,z]
        # train_cov.shape=[K,z,z]
        n_k = train_phi.shape[0]
        C = 0  # C= integral(p(x)*p(x)*dx
        if n_k == 1:
            mu_k = train_mu
            cov_k = 2 * train_cov
            from torch.distributions import MultivariateNormal
            from torch.distributions import multivariate_normal

            gmm = multivariate_normal.MultivariateNormal(mu_k, cov_k)
            # gmm=MultivariateNormal(mu_k,cov_k)
            C = gmm.log_prob(mu_k)  # return rqt(integral p(x)p(x)dx)
        return C
    # compute the CauchySchwarzDivergence of the RFS which does not depend on the data points(features)
    # this calculation will be carried out once and will be used during training and testing
    C = None
    train_phi = torch.tensor([1]).to(device)
    from utils.mvtec_utils import  compute_l2_norm
    if with_rank:
        if args.use_l2gmm == 1:
            # compute the log l_2 norm as follows
            C = compute_l2_norm(train_phi, train_mu, train_cov)
            if C.isinf():
                C = torch.tensor([0.1], dtype=float).to(device=device)
            if C.isnan():
                C = torch.tensor([0.1], dtype=float).to(device=device)
        else:
            C = compute_CS_div(train_phi, train_mu, train_cov)
            if C.isinf():
                C = torch.tensor([0.1], dtype=float).to(device=device)
            if C.isnan():
                C = torch.tensor([0.1], dtype=float).to(device=device)
    from torch.distributions import multivariate_normal
    gmm = multivariate_normal.MultivariateNormal(train_mu, train_cov)
    from scipy.spatial.distance import mahalanobis
    # #calcualt log likelihood and dist
    for batch in train_loader_b1:
         x = batch
         card = torch.tensor(len(x), dtype=float)
         x = torch.tensor(x)
         x = x.float().to(device)
         y = torch.tensor([0])
    #
    #     from scipy.spatial.distance import mahalanobis
    #     # the sample energy is - log
         cov_inv = np.linalg.inv(train_cov.cpu().numpy())
    #
         dist = torch.tensor([mahalanobis(sample.cpu().numpy(), train_mu.cpu().numpy(), cov_inv) for sample in x]).to(device)
         dist_log = torch.log(torch.tensor([mahalanobis(sample.cpu().numpy(), train_mu.cpu().numpy(), cov_inv) for sample in x])).to(device)
         dist_log_sq = torch.log(torch.tensor([mahalanobis(sample.cpu().numpy(), train_mu.cpu().numpy(), cov_inv)**2 for sample in x])).to(device)
         dist_sq = torch.tensor([mahalanobis(sample.cpu().numpy(), train_mu.cpu().numpy(), cov_inv)**2 for sample in x]).to(device)
         sample_energy_train = gmm.log_prob(x)
         sample_dist_train = dist
         # the card loss should aldistso be -log
         card_loss = card_mle.compute_loss(lamda_hat, card.float().to(device), with_NLL=False)
         card_dist_loss=torch.log((card-lamda_hat)**2)
         card_dist_loss=(card-lamda_hat)**2/card_std**2
    #     # final likelihood is  log_pXcar - Xcar * log(c) + sum(b)
    #     # sum the loglilihood of all smaples based on NAive log
         if with_card:
             if with_rank:
                 # this for log likelihood
                 # sample_energy=torch.sum(sample_energy,dim=0).view(-1)+card_loss+(torch.log(C)*card.float().to(device=device))
                 # this for log liklihiid
                 sample_energy_train = torch.sum(sample_energy_train, dim=0).view(-1) + card_loss.view(-1) - (torch.log(C) * card.float().to(device=device)).view(-1)
                 sample_dist_train = torch.sum(dist, dim=0).view(-1) + card_loss.view(-1) - (torch.log(C) * card.float().to(device=device)).view(-1)
                 sample_dist_v2_train = torch.sum(dist_sq, dim=0).view(-1) + card_dist_loss.view(-1)
                 sample_dist_log_train = torch.sum(dist_log, dim=0).view(-1) + card_dist_loss.view(-1)
                 sample_dist_log_sq_train = torch.sum(dist_log_sq, dim=0).view(-1) + card_dist_loss.view(-1)
                 card_sample_train = card_loss.view(-1) - (torch.log(C) * card.float().to(device=device)).view(-1)
             else:
                 # the log
                 sample_energy_train = torch.sum(sample_energy_train, dim=0).view(-1) + card_loss.view(-1) + (card.float().to(device=device) + 1).lgamma().view(-1)
                 card_sample_train = card_loss.view(-1) + (card.float().to(device=device) + 1).lgamma().view(-1)
                 sample_dist_train = torch.sum(dist, dim=0).view(-1) + card_loss.view(-1)+ (card.float().to(device=device) + 1).lgamma().view(-1)
                 sample_dist_v2_train = torch.sum(dist_sq, dim=0).view(-1) + card_dist_loss.view(-1)
                 sample_dist_log_train = torch.sum(dist_log, dim=0).view(-1) + card_dist_loss.view(-1)
                 sample_dist_log_sq_train = torch.sum(dist_log_sq, dim=0).view(-1) + card_dist_loss.view(-1)


    #            #
         else:
             sample_energy_train = torch.sum(sample_energy_train, dim=0).view(-1)
             card_sample_train = torch.tensor([1]).view(-1).to(device)
             sample_dist_train = torch.sum(dist, dim=0).view(-1)
             sample_dist_v2_train = torch.sum(dist_sq, dim=0).view(-1)
             sample_dist_log_train = torch.sum(dist_log, dim=0).view(-1)
             sample_dist_log_sq_train = torch.sum(dist_log_sq, dim=0).view(-1)
    #
         energy_train.append(sample_energy_train.detach().cpu())
         card_energy_train.append(card_sample_train.detach().cpu())
         dist_train.append(sample_dist_train.detach().cpu())
         dist_train_v2.append(sample_dist_v2_train.detach().cpu())
         dist_train_log.append(sample_dist_log_train.detach().cpu())
         dist_train_log_sq.append(sample_dist_log_sq_train.detach().cpu())
    #
         labels_train.append(y)
    energy_train = torch.cat(energy_train).numpy()
    card_energy_train = torch.cat(card_energy_train).numpy()
    dist_train = torch.cat(dist_train).numpy()
    dist_train_v2 = torch.cat(dist_train_v2).numpy()
    dist_train_log = torch.cat(dist_train_log).numpy()
    dist_train_log_sq = torch.cat(dist_train_log_sq).numpy()
    # label o:normal samples
    labels_train = torch.cat(labels_train).numpy()

    # Obtaining Labels and energy scores for test data
    energy_test = []
    energy_test2 = []
    energy_test3 = []
    labels_test = []
    card_energy_test = []
    dist_test = []
    dist_test_v2 = []
    dist_test_log = []
    dist_test_log_sq = []
    dist_test_final = []
    for i in range(len(dataloader_test)):
        x = dataloader_test[i]
        y = test_labels[i]
        card = torch.tensor(len(x), dtype=float).to(device)
        x = torch.tensor(x)
        x = x.float().to(device)
        cov_inv = np.linalg.inv(train_cov.detach().cpu().numpy())
        dist = torch.tensor([mahalanobis(sample.cpu().numpy(), train_mu.cpu().numpy(), cov_inv) for sample in x]).to(device)
        dist_log = torch.log(torch.tensor([mahalanobis(sample.cpu().numpy(), train_mu.cpu().numpy(), cov_inv) for sample in x])).to(device)

        dist_log_sq = torch.log(torch.tensor([mahalanobis(sample.cpu().numpy(), train_mu.cpu().numpy(), cov_inv)**2 for sample in x])).to(device)
        dist_sq = torch.tensor([mahalanobis(sample.cpu().numpy(), train_mu.cpu().numpy(), cov_inv)**2 for sample in x]).to(device)
        dist_negative_sq=dist_sq*0.5
        sample_energy = gmm.log_prob(x)
        sample_energy2 = torch.square(gmm.log_prob(x))
        #sample_energy3 =
        card_loss = card_mle.compute_loss(lamda_hat, card.float().to(device), with_NLL=False)
        card_loss2 = card_mle.compute_loss(lamda_hat, card.float().to(device), with_NLL=False)
        #negative cardinality distribution
        card_loss3 = card_mle.compute_loss(lamda_hat, card.float().to(device), with_NLL=True)
        card_dist_loss= (card.float()-lamda_hat).to(device)#: torch.Size([])

        # sum the loglilihood of all smaples based on NAive log
        if with_card:
            if with_rank:
                # this for log likelihood
                # sample_energy=torch.sum(sample_energy,dim=0).view(-1)+card_loss+(torch.log(C)*card.float().to(device=device))
                # this for log liklihiid
                sample_energy = torch.sum(sample_energy, dim=0).view(-1) + card_loss.view(-1) - (torch.log(C) * card.float().to(device=device)).view(-1)
                sample_energy2 = torch.sum(sample_energy2, dim=0).view(-1) + card_loss2.view(-1) - (
                            torch.log(C) * card.float().to(device=device)).view(-1)
                #sample_energy3 = torch.prod(sample_energy3, dim=0).view(-1)*card_loss3.view(-1)/(C* card.float().to(device=device)).view(-1)
                card_sample_test = card_loss.view(-1) - (torch.log(C) * card.float().to(device=device)).view(-1)
                sample_dist = torch.sum(dist, dim=0).view(-1) + card_loss.view(-1) - (torch.log(C) * card.float().to(device=device)).view(-1)
                sample_dist_v2 = torch.sum(dist_sq, dim=0).view(-1) + card_dist_loss.view(-1)
                sample_dist_log = torch.sum(dist_log, dim=0).view(-1) + card_dist_loss.view(-1)
                sample_dist_log_sq = torch.sum(dist_log_sq, dim=0).view(-1) + card_dist_loss.view(-1)
                sample_dist_final=torch.sum(dist_negative_sq, dim=0).view(-1) + card_loss3.view(-1) +(card.float().to(device=device)).view(-1)
            else:
                # sample_energy = torch.sum(sample_energy, dim=0).view(-1) + card_loss+
                # this for log like
                # sample_energy = torch.sum(sample_energy, dim=0).view(-1) + card_loss + (card.float().to(device=device)+1).lgamma()
                # the log
                sample_energy = torch.sum(sample_energy, dim=0).view(-1) + card_loss.view(-1) + (card.float().to(device=device) + 1).lgamma().view(-1)
                sample_energy2 = torch.sum(sample_energy2, dim=0).view(-1) + card_loss2.view(-1) + (
                            card.float().to(device=device) + 1).lgamma().view(-1)
                sample_dist_final = torch.sum(dist_negative_sq, dim=0).view(-1) + card_loss3.view(-1) -(card.float().to(device=device) + 1).lgamma().view(-1)

                #sample_energy3 = torch.prod(sample_energy3, dim=0).view(-1)* card_loss3.view(-1)*(card.float().to(device=device) + 1).lgamma().exp().view(-1)
                card_sample_test = card_loss.view(-1) + (card.float().to(device=device) + 1).lgamma().view(-1)
                sample_dist= torch.sum(dist, dim=0).view(-1) + card_loss.view(-1) + (card.float().to(device=device) + 1).lgamma().view(-1)
                sample_dist_v2 = torch.sum(dist_sq, dim=0).view(-1) + card_dist_loss.view(-1)
                sample_dist_log = torch.sum(dist_log, dim=0).view(-1) + card_dist_loss.view(-1)
                sample_dist_log_sq = torch.sum(dist_log_sq, dim=0).view(-1) + card_dist_loss.view(-1)

                #
        else:
            sample_energy = torch.sum(sample_energy, dim=0).view(-1)#torch.Size([1])
            sample_energy2 = torch.sum(sample_energy2, dim=0).view(-1)  # torch.Size([1])
            #sample_energy3 = torch.prod(sample_energy3, dim=0).view(-1)  # torch.Size([1])
            card_sample_test = card_dist_loss.view(-1)#torch.Size([1])
            sample_dist= torch.sum(dist, dim=0).view(-1)#torch.Size([1])
            sample_dist_v2 = torch.sum(dist_sq, dim=0).view(-1)#torch.Size([1])
            sample_dist_log = torch.sum(dist_log, dim=0).view(-1)#torch.Size([1])
            sample_dist_log_sq = torch.sum(dist_log_sq, dim=0).view(-1)#torch.Size([1])
            sample_dist_final = torch.sum(dist_negative_sq, dim=0).view(-1)


        labels_test.append(y)
        card_energy_test.append(card_sample_test.detach().cpu())
        energy_test.append(sample_energy.detach().cpu())
        energy_test2.append(sample_energy2.detach().cpu())
        #energy_test3.append(sample_energy3.detach().cpu())
        dist_test.append(sample_dist.detach().cpu())
        dist_test_v2.append(sample_dist_v2.detach().cpu())
        dist_test_log.append(sample_dist_log.detach().cpu())
        dist_test_log_sq.append(sample_dist_log_sq.detach().cpu())
        dist_test_final.append(sample_dist_final.detach().cpu())


    threshold = np.percentile(energy_train, pecent)
    pred = (energy_test > threshold).astype(int)

    energy_test = torch.cat(energy_test).numpy()
    energy_test2 = torch.cat(energy_test2).numpy()
    dist_test_final=torch.cat(dist_test_final).numpy()
    threshold = np.percentile(energy_train, pecent)
    energy_test3 = (energy_test > threshold).astype(int)
    print('sum of energy test',energy_test.sum())
    labels_test = torch.cat(labels_test).numpy()
    card_energy_test = torch.cat(card_energy_test).numpy()
    dist_test = torch.cat(dist_test).numpy()
    dist_test_v2 = torch.cat(dist_test_v2).numpy()
    dist_test_log = torch.cat(dist_test_log).numpy()
    dist_test_log_sq = torch.cat(dist_test_log_sq).numpy()
    from sklearn.metrics import roc_auc_score, confusion_matrix, auc, roc_curve
    log_like_score = roc_auc_score(labels_test,energy_test)*100
    log_like_score2 = roc_auc_score(labels_test, energy_test2) * 100
    fpr, tpr, thresholds = roc_curve(labels_test,  energy_test3, pos_label=1)
    log_like_score3 = auc(fpr, tpr) * 100
    card_only_score = roc_auc_score(labels_test, card_energy_test)*100
    dist_score = roc_auc_score(labels_test, dist_test)*100
    dist_v2_score = roc_auc_score(labels_test, dist_test_v2)*100
    dist_log_score = roc_auc_score(labels_test, dist_test_log)*100
    dist_log_sq_score = roc_auc_score(labels_test, dist_test_log_sq)*100
    dist_final_score =roc_auc_score(labels_test, dist_test_final)*100
    print('log_like_score    {:0.2f}'.format(log_like_score))
    print('log_like_score2    {:0.2f}'.format(log_like_score2))
    print('log_like_score3    {:0.2f}'.format(log_like_score3))
    print('card_only_score   {:0.2f}'.format( card_only_score))
    print('dist_score        {:0.2f}'.format(dist_score))
    print('dist_v2_score     {:0.2f}'.format( dist_v2_score))
    print('final distance    {:0.2f}'.format(dist_final_score))
    print('dist_log_score    {:0.2f}'.format(dist_log_score))
    print('dist_log_sq_score {:0.2f}'.format(dist_log_sq_score))

    return log_like_score, card_only_score, dist_score, dist_v2_score, dist_log_score, dist_log_sq_score, dist_final_score

def calculate_ospa_distance(input_list,args):
    import numpy as np
    d = []
    for i in range(len(input_list)):
        X=input_list[i]
        c=i+1
        for j in range(c,len(input_list)):
            Y = input_list[j]
            print(i,j)
            d.append(ospa(X,Y,c=20,p=2))
    return d










from scipy.optimize import linear_sum_assignment
import numpy as np
import numpy.linalg as lin
def _calculation(X, Y, c, p):
    def d_c(x, y, c):
        return min(c, lin.norm(x - y))

    m = len(X)
    n = len(Y)
    if m == 0 and n == 0:
        return 0, 0, 0

    if m > n:
        # swap
        X, Y = Y, X
        m, n = n, m

    card_dist = c ** p * (n - m)

    D = np.zeros((n, n))
    for i in range(m):
        for j in range(n):
            D[i, j] = d_c(X[i], Y[j], c) ** p
    D[m:, :] = c ** p
    row_ind, col_ind = linear_sum_assignment(D)
    local_dist = D[row_ind[:m], col_ind[:m]].sum()

    return local_dist, card_dist, n


def ospa(X, Y, c, p):
    """
    Calculates Optimal Subpattern Assignment (OSPA) metric, defined by Dominic Schuhmacher, Ba-Tuong Vo, and Ba-Ngu Vo
    in "A Consistent Metric for Performance Evaluation of Multi-Object Filters". This is implementation using Hungarian
    method.
    https://en.wikipedia.org/wiki/Hungarian_algorithm
    http://www.hungarianalgorithm.com/examplehungarianalgorithm.php
    :param X: set of ndarray vectors
    :param Y: set of ndarray vectors
    :param c: c>0 . "The cut-off parameter c determines the relative weighting of the penalties assigned to
    cardinality and localization errors. A value of c which corresponds to the magnitude
    of a typical localization error can be considered small and has the effect of emphasizing
    localization errors. A value of c which corresponds to the maximal distance between
    targets can be considered large and has the effect of emphasizing cardinality errors."
    from Bayesian Multiple Target Filtering Using Random Finite Sets, BA-NGU VO, BA-TUONG VO, AND DANIEL CLARK
    :param p: The order parameter p determines the sensitivity of the metric to outliers. p>=1
    :return:
    """
    local_dist, card_dist, n = _calculation(X, Y, c, p)
    if n == 0:
        return 0
    return (1 / n * (local_dist + card_dist)) ** (1 / p)


def ospa_local_card(X, Y, c, p):
    local_dist, card_dist, n = _calculation(X, Y, c, p)
    if n == 0:
        return 0, 0
    return (1 / n * local_dist) ** (1 / p), (1 / n * card_dist) ** (1 / p)


def ospa_all(X, Y, c, p):
    local_dist, card_dist, n = _calculation(X, Y, c, p)
    if n == 0:
        return 0, 0, 0
    return (1 / n * (local_dist + card_dist)) ** (1 / p), (1 / n * local_dist) ** (1 / p), (1 / n * card_dist) ** (
            1 / p)
            
def describe_opencv(model,
                    img,
                    kpts,
                    patch_size=32,
                    mag_factor=3,
                    use_gpu=True):
    """
        Rectifies patches around openCV keypoints, and returns patches tensor
    """
    patches = []
    for kp in kpts:
        x, y = kp.pt
        s = kp.size
        a = kp.angle

        s = mag_factor * s / patch_size
        cos = math.cos(a * math.pi / 180.0)
        sin = math.sin(a * math.pi / 180.0)

        M = np.matrix(
            [[+s * cos, -s * sin, (-s * cos + s * sin) * patch_size / 2.0 + x],
             [+s * sin, +s * cos,
              (-s * sin - s * cos) * patch_size / 2.0 + y]])

        patch = cv2.warpAffine(
            img,
            M, (patch_size, patch_size),
            flags=cv2.WARP_INVERSE_MAP + cv2.INTER_CUBIC +
            cv2.WARP_FILL_OUTLIERS)

        patches.append(patch)

    patches = torch.from_numpy(np.asarray(patches)).float()
    patches = torch.unsqueeze(patches, 1)
    if use_gpu:
        patches = patches.cuda()
    descrs = model(patches)
    return descrs.detach().cpu().numpy()


def detetect_keypoint_desc(image,path,args,with_desc=False):

    if args.detector_type == 'sift-orb':
        kp_detector1 = cv2.SIFT_create()
        kp_detector2 = cv2.ORB_create(edgeThreshold=20, patchSize=16)
        kpt, desc = kp_detector1.detectAndCompute(image, None)
        kpt2, desc2 = kp_detector2.detectAndCompute(image, None)
        if with_desc:
            return kpt, desc,kpt2, desc2
        else:
            return kpt, kpt2
    elif args.detector_type == 'sift-kaze':
        kp_detector1 = cv2.SIFT_create()
        kp_detector2 = cv2.KAZE_create()
        kpt, desc = kp_detector1.detectAndCompute(image, None)
        kpt2, desc2 = kp_detector2.detectAndCompute(image, None)
        if with_desc:
            return kpt, desc,kpt2, desc2
        else:
            return kpt, kpt2
    elif args.detector_type == 'sift-akaze':
        kp_detector1 = cv2.SIFT_create()
        kp_detector2 = cv2.AKAZE_create()
        kpt, desc = kp_detector1.detectAndCompute(image, None)
        kpt2, desc2 = kp_detector2.detectAndCompute(image, None)
        if with_desc:
            return kpt, desc, kpt2, desc2
        else:
            return kpt, kpt2
    elif args.detector_type == 'sift':
        kp_detector = cv2.SIFT_create(edgeThreshold=args.sift_edgeThreshold)

    elif args.detector_type == 'orb':
        kp_detector = cv2.ORB_create(edgeThreshold=20, patchSize=16)

    elif args.detector_type == 'akaze':
        kp_detector = cv2.AKAZE_create()
    elif args.detector_type == 'kaze':
        kp_detector = cv2.KAZE_create()
    elif args.detector_type == 'mser':
        kp_detector = cv2.MSER_create()
        sift= cv2.SIFT_create()
        kp=kp_detector.detect(image)
        kpt,desc=sift.compute(image, kp)
        return kpt,desc
    elif args.detector_type == 'surf':
        #kp_detector = cv2.xfeatures2d.SURF_create(400)
        kp_detector = cv2.FastFeatureDetector_create()
        kp=kp_detector.detect(image,None)
        sift=cv2.SIFT_create()
        desc=sift.compute(image,kp)
        return kp, desc[1]
    elif args.detector_type == 'Harris':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp_detector=cv2.cornerHarris(gray , 2, 3, 0.04)
        dst = cv2.dilate(kp_detector, None)
    elif args.detector_type == 'sp':
        return detect_superpoint_descr(path, args, with_desc=False)
    elif args.detector_type == 'sp-with_sift':
        kp_detector = cv2.SIFT_create()
        args.detector_type = 'sp'
        kpt,desc=detect_superpoint_descr(path, args, with_desc=False)
        kp = list(kpt)
        sift_kp = [cv2.KeyPoint(x[0], x[1], 1) for x in kp]
        kp, desc2 = kp_detector.compute(image, sift_kp)
        args.detector_type = 'sp-with_sift'

        return kpt, desc2
    elif args.detector_type == 'gp':

        return detect_superpoint_descr(path, args, with_desc=False)


    # imgplot = plt.imshow(image)
    # plt.show()
    # detect keypoint
    # kpt=sift.detect(image)
    kpt, desc = kp_detector.detectAndCompute(image, None)
    if with_desc:
        return kpt, desc,
    else:
        return kpt

def detect_superpoint_descr(image,args,with_desc=False):
    device = 'cuda'
    #import all required libs
    import os
    import sys
    import numpy
    sys.path.insert(1,'/media/SSD2/DATA/ammar/Industrial_image_anomaly/image-matching/fem/')
    from util import calculate_h, swap_rows, project_points2
    from wrapper import SuperPoint
    import drawing
    import imageio
    import torch
    from superpoint_magicleap import PointTracker
    from goodpoint import GoodPoint
    from nonmaximum import MagicNMS
    from bench import get_points_desc, preprocess, draw_matches, replication_ratio, coverage, harmonic
    import cv2
    #pretrained goodpoint
    weight = "/media/SSD2/DATA/ammar/Industrial_image_anomaly/image-matching/fem/snapshots/super3400.pt"
    sp_path = '/media/SSD2/DATA/ammar/Industrial_image_anomaly/image-matching/fem/superpoint_magicleap/superpoint_v1.pth'
    nms = MagicNMS(nms_dist=8)
    sp = SuperPoint(MagicNMS()).to(device).eval()
    sp.load_state_dict(torch.load(sp_path))

    gp = GoodPoint(dustbin=0,activation=torch.nn.LeakyReLU(),
                   batchnorm=True,
                   grid_size=8,
                   nms=nms).eval().to(device)

    gp.load_state_dict(torch.load(weight, map_location=device)['superpoint'])
    img = imageio.imread(image[0], pilmode='L')#shape(H,W)
    from PIL import Image as Pil_im
    from torchvision import transforms as T
    PIL_image = Pil_im.fromarray(img)
    Resize_torch = T.Resize(256)
    crop_center_torch = T.CenterCrop(224)
    # crop_center_torch = T.CenterCrop(256)
    croped_img = crop_center_torch(Resize_torch(PIL_image))
    image = np.array(croped_img)
    resize = Resize((224, 224))
    # cv2.imshow('img1', img)
    img = resize(img).squeeze()#shape(224,224)
    super_thresh = 0.015
    # conf_thresh = 0.0207190856295525# AUC SP:0.681
    #conf_thresh = 0.1
    conf_thresh=args.sp_thresh
    timg1 = numpy.expand_dims(numpy.expand_dims(img.astype('float32'), axis=0), axis=0)#shape(1,1,224,224)
    with torch.no_grad():
        if args.detector_type == 'gp':
            gp.eval()
            pts_1, desc_1_ = gp.points_desc(torch.from_numpy(timg1).to(next(gp.parameters())), threshold=conf_thresh)
        if args.detector_type=='sp':
            sp.eval()
            pts_1, desc_1_ = sp.points_desc(torch.from_numpy(timg1).to(next(sp.parameters())), threshold=conf_thresh)
    return pts_1, numpy.array(desc_1_[0].detach().cpu())
import numpy
def resize(img, size):

    if len(img.shape) == 3 and numpy.argmin(img.shape) == 0:
        img = img.transpose(1, 2, 0)
    if len(img.shape) == 3 and img.shape[2] != 1:
        return cv2.resize(img, size)
    return cv2.resize(img, size)[numpy.newaxis, :]
class Resize:
    def __init__(self, size, keep_ratio=False):
        self.size = size
        self.keep_ratio = keep_ratio

    def __call__(self, img, return_size=False):
        data_res, size = self.resize_data(img)
        if return_size:
            return data_res, size
        return data_res

    def resize_data(self, img):
        if self.keep_ratio:
            assert numpy.argmin(img.shape) == 2 or len(img.shape) == 2
            if img.shape[0] / self.size[0] != img.shape[1] / self.size[1]:
                main_axis = numpy.argmin(img.shape[:-1])
                ratio = img.shape[main_axis] / self.size[main_axis]
                new_size = round(img.shape[1] / ratio), round(img.shape[0] / ratio)
                return resize(img, new_size), new_size
        return resize(img, self.size), self.size

def detect_d2_net_descr(path,args):
    device = 'cuda'
    #import all required libs
    import os
    import sys
    sys.path.insert(1,'/media/SSD2/DATA/ammar/Industrial_image_anomaly/d2-net/')
    import argparse
    import numpy as np
    import imageio
    from torchvision import transforms as T
    from skimage.transform import resize as skresize
    import torch
    from tqdm import tqdm
    import scipy
    import scipy.io
    import scipy.misc
    import os
    from lib.model_test import D2Net
    from lib.utils import preprocess_image
    from lib.pyramid import process_multiscale
    # Creating CNN model
    use_cuda = torch.cuda.is_available()
    #args.model_file='/media/SSD2/DATA/ammar/Industrial_image_anomaly/d2-net/models/d2_tf_no_phototourism.pth'
    #args.model_file = '/media/SSD2/DATA/ammar/Industrial_image_anomaly/d2-net/models/d2_tf.pth'
    args.model_file = '/media/SSD2/DATA/ammar/Industrial_image_anomaly/d2-net/models/d2_ots.pth'
    args.preprocessing = 'caffe'
    args.max_edge = 1600
    args.max_sum_edges = 2800
    args.use_relu= 'use_relu'
    args.multiscale = 1
    model = D2Net(model_file=args.model_file,use_relu=args.use_relu,use_cuda=use_cuda)
    model.detection.edge_threshold = 5
    # Process the file
    import matplotlib.pyplot as plt
    from PIL import Image as Pil_im
    image = imageio.imread(path)#shape(H,W,C)
    #covert to Pil image
    PIL_image = Pil_im.fromarray(image)
    Resize_torch=T.Resize(256)
    crop_center_torch=T.CenterCrop(224)
    #crop_center_torch = T.CenterCrop(256)
    croped_img=crop_center_torch(Resize_torch(PIL_image))
    image=np.array(croped_img)#image.shape=(224,224,3)
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
        image = np.repeat(image, 3, -1)

        # TODO: switch to PIL.Image due to deprecation of scipy.misc.imresize.
    resized_image = image
    if max(resized_image.shape) > args.max_edge:
            # resized_image = scipy.misc.imresize(
            #    resized_image,
            #    args.max_edge / max(resized_image.shape)
            # ).astype('float')
        resized_image = skresize(resized_image, (args.max_edge, args.max_edge))
    if sum(resized_image.shape[: 2]) > args.max_sum_edges:
        resized_image = scipy.misc.imresize(resized_image,args.max_sum_edges / sum(resized_image.shape[: 2])).astype('float')

    fact_i = image.shape[0] / resized_image.shape[0]
    fact_j = image.shape[1] / resized_image.shape[1]

    input_image = preprocess_image(resized_image,preprocessing=args.preprocessing)
    with torch.no_grad():
        if args.multiscale:
            keypoints, scores, descriptors = process_multiscale(torch.tensor(input_image[np.newaxis, :, :, :].astype(np.float32),device=device),model)
        else:
            keypoints, scores, descriptors = process_multiscale(torch.tensor(input_image[np.newaxis, :, :, :].astype(np.float32),device=device),model,scales=[1])

        # Input image coordinates
    keypoints[:, 0] *= fact_i
    keypoints[:, 1] *= fact_j
        # i, j -> u, v
    keypoints = keypoints[:, [1, 0, 2]]

    if args.retrun_image:
        return keypoints,scores,descriptors,resized_image

    return keypoints,scores,descriptors

