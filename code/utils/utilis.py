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


def Possion_UnnorLoglike(lamda, card, with_NLL=False):
    Loglike = card * torch.log(lamda) - (card + 1).lgamma()
    # Loglike=card*torch.log(lamda)
    if not with_NLL:
        return Loglike
    if with_NLL:
        return -Loglike

    
class PossionMLE:
    """MLE class for cardinality."""
    def __init__(self, args, card_loader, device,plot_dis=False):
        self.args = args
        self.card_train_loader= card_loader
       # _, _, self.card_train_loader= data
        self.device = device
        self.plot_distribution=plot_dis
    def compute_loss(self,lamda_hat,x, with_NLL):
        poss=torch.distributions.poisson.Poisson(lamda_hat)
        if with_NLL:
            card_loss= - poss.log_prob(x)
        else:
            card_loss = poss.log_prob(x)
        return  card_loss

    def comput_lamda(self):

        for i,card_data in enumerate(self.card_train_loader):
            if self.args.fewshots:
                card_data=card_data[0:self.args.fewshots_exm]
            lamda_hat=torch.mean(card_data.float().to(self.device))
            if self.plot_distribution==1:
                card_mat = card_data.cpu().numpy()
                import matplotlib.pyplot as plt
                plt.hist(card_mat,card_mat.shape[0]//2)
                plt.title(self.args.catogery_name+' cardinality histogram of training samples ')
                plt.show()

        return lamda_hat
