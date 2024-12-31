import math
import cv2
import json
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

# from torch.autograd import Variable
from copy import deepcopy
from sklearn import metrics
from skimage import filters
import numpy.matlib

# import statsmodels.api as sm
# from lmfit import Parameters, minimize
from scipy.optimize import minimize as minimize_sci
from scipy.spatial.distance import directed_hausdorff


class NpEncoder(json.JSONEncoder):
    '''
    json encoder 专用
    '''

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):  # add this line
            return obj.tolist()  # add this line
        return json.JSONEncoder.default(self, obj)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def dice_coef(self, y_pred, y_true):
        pred_probs = torch.sigmoid(y_pred)
        y_true_f = y_true.view(-1)
        y_pred_f = pred_probs.view(-1)
        intersection = torch.sum(y_true_f * y_pred_f)
        return (2.0 * intersection + self.smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + self.smooth)

    def forward(self, y_pred, y_true):
        return -self.dice_coef(y_pred, y_true)


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()

    def forward(self, y_pred, y_true):
        return -torch.mean(torch.sum(y_true * torch.log(F.softmax(y_pred, dim=1)), dim=1))


class WCELoss(nn.Module):
    def __init__(self):
        super(WCELoss, self).__init__()

    def forward(self, y_pred, y_true, weights):
        y_true = y_true / (y_true.sum(2).sum(2, dtype=torch.float).unsqueeze(-1).unsqueeze(-1))
        y_true[y_true != y_true] = 0.0
        y_true = torch.sum(y_true, dim=1, dtype=torch.float).unsqueeze(1)
        y_true = y_true * weights.to(torch.float)
        old_range = torch.max(y_true) - torch.min(y_true)
        new_range = 100 - 1
        y_true = (((y_true - torch.min(y_true)) * new_range) / old_range) + 1
        return -torch.mean(torch.sum(y_true * torch.log(F.softmax(y_pred, dim=1)), dim=1))


class WMSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, y_pred, y_true, weight):
        return torch.mean(torch.pow(y_pred - y_true, 2) * weight)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, window_sigma=1.5, channel=1, size_average=True) -> None:
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.window = self.create_window(window_size, channel, window_sigma).cuda()
        self.C1 = 0.01**2
        self.C2 = 0.03**2

    def create_window(self, window_size, channel, sigma=1.5):
        gauss = torch.Tensor([math.exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)])
        _1D_window = gauss / gauss.sum()
        _1D_window = _1D_window.unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = torch.autograd.Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    def forward(self, img1, img2):
        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=self.channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))
        if self.size_average:
            ssim_map_mean = ssim_map.mean()
        else:
            ssim_map_mean = ssim_map.mean(1).mean(1).mean(1)
        return ssim_map_mean


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * torch.autograd.Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def distillation(out_s, out_t, labels, temperature, alpha):
    p = F.log_softmax(out_s / temperature, dim=1)
    q = F.softmax(out_t / temperature, dim=1)
    loss_kl = F.kl_div(p, q, size_average=False) * (temperature**2) / out_s.shape[0]
    loss_ce = F.cross_entropy(out_s, labels)
    return loss_kl * alpha + loss_ce * (1 - alpha)


def kl_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)


def KL_devergence(p, q):
    """
    Calculate the KL-divergence of (p,q)
    :param p:
    :param q:
    :return:
    """
    q = torch.nn.functional.softmax(q, dim=0)
    q = (
        torch.sum(q, dim=0) / p.shape[0]
    )  # dim:缩减的维度,q的第一维是batch维,即大小为batch_size大小,此处是将第j个神经元在batch_size个输入下所有的输出取平均
    s1 = torch.sum(p * torch.log(p / q))
    s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))
    return s1 + s2


def active_contour_loss(y_pred, y_true, weight=10):
    '''
    y_true, y_pred: tensor of shape (B, C, H, W), where y_true[:,:,region_in_contour] == 1, y_true[:,:,region_out_contour] == 0.
    weight: scalar, length term weight.
    '''
    # length term
    delta_r = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]  # horizontal gradient (B, C, H-1, W)
    delta_c = y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]  # vertical gradient   (B, C, H,   W-1)

    delta_r = delta_r[:, :, 1:, :-2] ** 2  # (B, C, H-2, W-2)
    delta_c = delta_c[:, :, :-2, 1:] ** 2  # (B, C, H-2, W-2)
    delta_pred = torch.abs(delta_r + delta_c)

    epsilon = 1e-8  # where is a parameter to avoid square root is zero in practice.
    lenth = torch.mean(torch.sqrt(delta_pred + epsilon))  # eq.(11) in the paper, mean is used instead of sum.

    # 区域
    c_in = torch.ones_like(y_pred)
    c_out = torch.zeros_like(y_pred)

    region_in = torch.mean(y_pred * (y_true - c_in) ** 2)  # equ.(12) in the paper, mean is used instead of sum.
    region_out = torch.mean((1 - y_pred) * (y_true - c_out) ** 2)
    region = region_in + region_out

    loss = weight * lenth + region

    return loss


class WingLoss(nn.Module):
    def __init__(self, omega=10, epsilon=2):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon

    def forward(self, pred, target):
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.omega]
        delta_y2 = delta_y[delta_y >= self.omega]
        loss1 = self.omega * torch.log(1 + delta_y1 / self.epsilon)
        C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
        loss2 = delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))


# def numeric_score(prediction, groundtruth):
#     """Computation of statistical numerical scores:
#     * FP = False Positives
#     * FN = False Negatives
#     * TP = True Positives
#     * TN = True Negatives
#     return: tuple (FP, FN, TP, TN)
#     """
#     FP = np.float(np.sum((prediction == 1) & (groundtruth == 0)))
#     FN = np.float(np.sum((prediction == 0) & (groundtruth == 1)))
#     TP = np.float(np.sum((prediction == 1) & (groundtruth == 1)))
#     TN = np.float(np.sum((prediction == 0) & (groundtruth == 0)))
#     return FP, FN, TP, TN

# def accuracy_score(prediction, groundtruth):
#     FP, FN, TP, TN = numeric_score(prediction, groundtruth)
#     N = FP + FN + TP + TN
#     accuracy = np.divide(TP + TN, N + 1e-10)
#     return accuracy# * 100.0


def average_distance(shape_gt, shape):
    '''
    计算两个shape对应点之间的距离
    shape: [N, 2] numpy
    pred_shape: [N, 2] numpy
    '''
    return metrics.mean_squared_error(shape_gt, shape)


def hausdorff_distance(boundary_gt, boundary):
    '''
    计算两个boundary点集的双向HD距离
    boundary_gt: [N, 2] numpy
    boundary: [M, 2] numpy
    '''
    return max(directed_hausdorff(boundary_gt, boundary)[0], directed_hausdorff(boundary, boundary_gt)[0])


def shape_to_hausdorff_distance(shape_gt, shape, WH):
    '''
    先将shape、shape_gt变为mask，然后提取边缘，计算hausdorff distance
    '''
    boundary = cv2.Canny(shape_to_mask(shape, WH), 0.3, 0.7)
    boundary_gt = cv2.Canny(shape_to_mask(shape_gt, WH), 0.3, 0.7)
    return max(directed_hausdorff(boundary, boundary_gt)[0], directed_hausdorff(boundary_gt, boundary)[0])


def shape_to_dice_np(shape, label):
    '''
    只计算dice，pred可以是prob、binary
    '''
    pred = shape_to_mask(shape, (label.shape[1], label.shape[0]))
    if len(np.unique(pred)) > 2:
        threshold = filters.threshold_otsu(pred, nbins=256)
        pred = np.where(pred > threshold, 1, 0)
    pred = pred.reshape(-1)
    label = label.reshape(-1)
    interaction = np.sum(pred * label)
    dice = (2 * interaction) / (np.sum(label) + np.sum(pred))
    return dice


def shape_to_dice_tensor(shape, label):
    '''
    只计算dice，pred可以是prob、binary
    shape: [61/51, 2]
    label: [B, H, W]
    '''
    pred = shape_to_mask(shape, (label.shape[-1], label.shape[-2]))
    if len(np.unique(pred)) > 2:
        threshold = filters.threshold_otsu(pred, nbins=256)
        pred = np.where(pred > threshold, 1, 0)
    pred = torch.from_numpy(pred)
    pred = pred.reshape(-1)
    label = label.reshape(-1)
    interaction = torch.sum(pred * label)
    dice = (2 * interaction) / (torch.sum(label) + torch.sum(pred))
    return dice


def dice_score(pred, label):
    '''
    pred可以是prob、binary
    '''
    if pred.shape[1] == 2:  # shape
        pred = shape_to_mask(pred, (label.shape[1], label.shape[0]))
    else:  # 2D-image
        if len(np.unique(pred)) > 2:  # image-continous
            threshold = filters.threshold_otsu(pred, nbins=256)
            pred = np.where(pred > threshold, 1, 0)
    pred = pred.reshape(-1)
    label = label.reshape(-1)
    interaction = np.sum(pred * label)
    dice = (2 * interaction) / (np.sum(label) + np.sum(pred))
    return dice


def AUC_score(SR, GT, threshold=0.5):
    GT = GT.ravel()  # we want to make them into vectors
    SR = SR.ravel()  # .detach()
    roc_auc = metrics.roc_auc_score(GT, SR)
    return roc_auc


# def recall_score(prediction, groundtruth):
#     # TPR, sensitivity
#     # prediction = prediction.detach().cpu().numpy()
#     # groundtruth = groundtruth.detach().cpu().numpy()
#     FP, FN, TP, TN = numeric_score(prediction, groundtruth)
#     # print(FP, FN, TP, TN)
#     if (TP + FN) <= 0.0:
#         return 0.0
#     TPR = np.divide(TP, TP + FN + 1e-10)
#     return TPR# * 100.0

# def specificity_score(prediction, groundtruth):
#     FP, FN, TP, TN = numeric_score(prediction, groundtruth)
#     if (TN + FP) <= 0.0:
#         return 0.0
#     TNR = np.divide(TN, TN + FP + 1e-10)
#     return TNR# * 100.0

# def fdr_score(prediction, groundtruth):
#     FP, FN, TP, TN = numeric_score(prediction, groundtruth)
#     fdr = FP / (FP + TP)
#     return  fdr

# def intersection_over_union(prediction, groundtruth):
#     FP, FN, TP, TN = numeric_score(prediction, groundtruth)
#     if (TP + FP + FN) <= 0.0:
#         return 0.0
#     return TP / (TP + FP + FN)# * 100.0


def all_score(prediction, groundtruth, mask=None):
    # auc
    auc = AUC_score(prediction, groundtruth)
    if len(np.unique(prediction)) > 1:
        threshold = filters.threshold_otsu(prediction, nbins=256)
    else:
        threshold = 0.5
    outputs = np.where(prediction > threshold, 1, 0)
    if mask is None:
        FP = np.sum((outputs == 1) & (groundtruth == 0))
        FN = np.sum((outputs == 0) & (groundtruth == 1))
        TP = np.sum((outputs == 1) & (groundtruth == 1))
        TN = np.sum((outputs == 0) & (groundtruth == 0))
    else:
        FP = np.sum((outputs == 1) & (groundtruth == 0) & (mask == 1.0))
        FN = np.sum((outputs == 0) & (groundtruth == 1) & (mask == 1.0))
        TP = np.sum((outputs == 1) & (groundtruth == 1) & (mask == 1.0))
        TN = np.sum((outputs == 0) & (groundtruth == 0) & (mask == 1.0))
    total = FP + FN + TP + TN
    # acc
    acc = np.divide(TP + TN, total + 1e-10)

    # sen or TPR or recall
    # if (TP + FN) <= 0.0:
    #     return 0.0
    sen = np.divide(TP, TP + FN + 1e-10)

    # spe or TNR
    # if (TN + FP) <= 0.0:
    #     return 0.0
    spe = np.divide(TN, TN + FP + 1e-10)

    # # fdr
    # fdr = np.divide(FP, FP + TP + 1e-10)
    # # FP / (FP + TP+ 1e-10)

    # precision
    pre = np.divide(TP, TP + FP + 1e-10)

    # f1=dice
    # f1 = np.divide(2 * pre * sen, pre + sen + 1e-10)

    # IOU
    # if (TP + FP + FN) <= 0.0:
    #     return 0.0
    # iou = TP / (TP + FP + FN)# * 100.0
    iou = np.divide(TP, TP + FP + FN + 1e-10)

    # DC: Dice Coefficient | JS: Jaccard similarity
    # outputs = prediction > 0.5
    # masks = groundtruth == np.max(groundtruth)
    # inter = np.sum(outputs*masks)
    # dc = 2*inter / (np.sum(outputs)+np.sum(masks))
    dc = 2 * TP / (2 * TP + FN + FP + 1e-10)
    # js = inter / (np.sum(outputs+masks))

    # kappa
    # po = (TP + TN)/(total)  # acc
    # pe = ((FN + FP) * (FN + TN) + (TN + TP) * (FP + TP)) / (total * total)
    # kappa = (acc - pe) / (1 - pe)

    # return acc, auc, sen, spe, f1, fdr, pre, dc, js, iou
    return acc, auc, sen, spe, pre, iou, dc


def iou_score(prediction, groundtruth):
    '''
    二值化的pred
    '''
    FP = np.sum((prediction == 1) & (groundtruth == 0))
    FN = np.sum((prediction == 0) & (groundtruth == 1))
    TP = np.sum((prediction == 1) & (groundtruth == 1))
    TN = np.sum((prediction == 0) & (groundtruth == 0))
    iou = np.divide(TP, TP + FP + FN + 1e-10)
    return iou


def compute_iou(predictions, masks):
    """
    计算单张图片的IOU
    compute IOU between two combined masks, this does not follow kaggle's evaluation
    :return: IOU, between 0 and 1
    """
    # ious = []
    # for i in range(0, len(img_ids)):
    #     pred = predictions[i]
    #     img_id = img_ids[i]
    #     mask_path = os.path.join(Option.root_dir, img_id, 'mask.png')
    #     mask = np.asarray(Image.open(mask_path).convert('L'), dtype=np.bool)
    #     union = np.sum(np.logical_or(mask, pred))
    #     intersection = np.sum(np.logical_and(mask, pred))
    #     iou = intersection/union
    #     ious.append(iou)
    # df = pd.DataFrame({'img_id':img_ids,'iou':ious})
    # df.to_csv('IOU.csv', index=False)

    union = np.sum(np.logical_or(predictions, masks))
    intersection = np.sum(np.logical_and(predictions, masks))
    iou = intersection / union
    return iou


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)


# about KD

# def attention_transfer(x):
#     return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

# def at_loss(s, t):
#     # return [(attention_transfer(i) - attention_transfer(j)).pow(2).mean() for i, j in zip(s, t)]
#     return (attention_transfer(s) - attention_transfer(t)).pow(2).mean()

# def at_loss(s, t):
#     ''' [batch_size, C, H, W]'''
#     norm_t = F.normalize(t.pow(2).mean(1).view(t.size(0), -1))
#     norm_s = F.normalize(s.pow(2).mean(1).view(s.size(0), -1))
#     return (norm_t - norm_s).pow(2).mean()

# def distillation(out_s, out_t, labels, temperature=4, alpha=0.9):
#     p = F.log_softmax(out_s/temperature, dim=1)
#     q = F.softmax(out_t/temperature, dim=1)
#     loss_kl = F.kl_div(p, q, size_average=False) * (temperature ** 2) / out_s.shape[0]
#     loss_ce = F.cross_entropy(out_s, labels)
#     return loss_kl * alpha + loss_ce * (1-alpha)


def getPerspectiveTransform_pts_tensor(mean_shape, pts_pred, pts_idx=[0, 16, 32, 48]):
    '''
    根据pts_pred获得透视变换矩阵，将mean_shape映射过去。
    mean_shape: [1, 1, 49, 2]
    pts_pred: [4, 2]
    '''
    mean_shape = torch.squeeze(mean_shape)  # [49, 2]
    mean_shape_np = mean_shape.numpy()
    pts_pred = pts_pred.numpy()
    prior_pt = mean_shape_np[pts_idx, :]  # [4, 2]
    prior_pt = prior_pt.astype(np.float32)  # [4, 2]
    pts_pred = pts_pred.astype(np.float32)  # [4, 2]
    mtx = cv2.getPerspectiveTransform(prior_pt, pts_pred).astype(np.float32)  # [3, 3]

    ones = np.ones((mean_shape_np.shape[0], 1), dtype=np.float32)  # [49, 1]
    prior_ones = np.concatenate((mean_shape_np, ones), axis=1)  # [49, 3]
    pos_shape_np = np.matmul(prior_ones, mtx.T)  # [49, 3]
    w = np.expand_dims(pos_shape_np[:, -1], axis=1).repeat(2, axis=1)  # [49, 2]
    pos_shape_np = pos_shape_np[:, :2] / w  # [49, 2]
    # pos_shape = pos_shape_np
    pos_shape = torch.from_numpy(pos_shape_np)
    return pos_shape


def argmax_tensor(tensor, return_value=False):
    '''
    求每个channel的最大值索引
    tensor: [batch, C, H, W]
    '''
    tensor = tensor.detach().cpu().squeeze(0)  # [C, H, W]
    flatten = tensor.view(tensor.shape[0], -1)
    flattern_argmax = flatten.argmax(1).view(-1, 1)
    result = torch.cat((flattern_argmax % tensor.shape[-1], torch.div(flattern_argmax, tensor.shape[-1], rounding_mode='trunc')), dim=1)
    if return_value:
        prob = flatten[flattern_argmax]
        return result, prob  # [4, 2], [4,]
    else:
        return result  # [4, 2]


def argmax_np_pre(maps):
    '''
    或许maps里单个通道的最大索引
    maps: [49, H, W], np.float32
    return: 每个channel最大值的坐标，及其值
    '''
    flatten = maps.reshape((maps.shape[0], -1))  # [49, HW]
    flatten_argmax = flatten.argmax(1).reshape((-1, 1))  # [49, 1]
    result = np.column_stack((flatten_argmax % maps.shape[-1], flatten_argmax // maps.shape[-1]))
    # value = [flatten[i, flatten_argmax[i]] for i in range(flatten.shape[0])]
    return result.astype(np.float32)  # , np.array(value)


def argmax_np(maps):
    '''
    或许maps里单个通道的最大索引
    maps: [H, W], np.float32
    return: 最大值的坐标，及其值
    '''
    flatten = maps.reshape(-1)  # [HW]
    flatten_argmax = flatten.argmax()
    result = np.column_stack((flatten_argmax % maps.shape[-1], flatten_argmax // maps.shape[-1]))[0]  # [W, H]
    value = flatten[flatten_argmax]
    # value = np.expand_dims(np.array(value), 1)
    return result, value


def soft_argmax_np(maps, scale=1):
    '''
    maps: [49, H, W], np
    return: 每个channel最大值的坐标，及其值
    '''
    C, H, W = maps.shape
    flatten = maps.reshape(C, -1)  # [C, H*W]
    # flatten *= scale  # 扩大值之间的差距
    # flatten = np.exp((flatten*scale))
    # flatten = (flatten*scale)**2
    for i in range(C):
        flatten[i] = np.interp(flatten[i], (flatten[i].min(), flatten[i].max()), (0, 2))
        flatten[i] = flatten[i] ** 2
        flatten[i] = flatten[i] / (np.sum(flatten[i]))
    idx_W, idx_H = np.meshgrid(np.arange(W), np.arange(H))
    idx_Ws = idx_W.reshape((-1, H * W))
    idx_Hs = idx_H.reshape((-1, H * W))
    result_Ws = np.sum(flatten * idx_Ws, axis=1)
    result_Hs = np.sum(flatten * idx_Hs, axis=1)
    results = np.column_stack([result_Ws, result_Hs])
    results_int = np.around(results).astype(np.int32)
    values = []
    for i in range(results_int.shape[0]):
        values.append(maps[i, results_int[i, 1], results_int[i, 0]])
    values = np.expand_dims(np.array(values), 1)
    return results_int, values


def soft_argmax_tensor(tensor, scale=1e2):
    '''
    求每个channel的期望
    tensor: [batch, C, H, W]
    return：真实坐标[W, H]
    '''
    # tensor = tensor.detach()  #.squeeze(0)  # [C, H, W]
    B, C, H, W = tensor.shape
    tensor = tensor.reshape(B, C, -1) * scale  # [C, H*W]
    tensor = tensor**2
    tensor = torch.softmax(tensor, dim=-1)
    tensor = tensor / torch.sum(tensor, dim=-1, keepdim=True)
    idx_H, idx_W = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
    idx_H = idx_H.to(tensor.device)
    idx_W = idx_W.to(tensor.device)
    idx_Hs = idx_H.reshape((-1, H * W))  # [1, H*W]
    idx_Ws = idx_W.reshape((-1, H * W))  # [1, H*W]
    result_H = torch.sum(tensor * idx_Ws * (W), -1) / W
    result_W = torch.sum(tensor * idx_Hs * (H), -1) / H
    result = torch.stack([result_W, result_H], -1)
    return result


def get_iou_pre(pts, label, input_WH=(544, 736)):
    '''
    计算每层outputs的最大值坐标，画封闭曲线，算iou
    pts: [49, 2] np.float32 resized_WH
    label: [H, W]
    '''
    # init_W, init_H = self.config.initial_size
    # label = label.squeeze().numpy()     # [H, W]
    W, H = input_WH
    H_ratio = label.shape[0] / H
    W_ratio = label.shape[1] / W
    scale = np.array([[W_ratio, 0], [0, H_ratio]])
    pts = np.dot(pts, scale)  # [49, 2] WH
    pts_int = np.int32(pts)
    pts_int[:, 0] = pts_int[:, 0].clip(min=0, max=label.shape[1] - 1)
    pts_int[:, 1] = pts_int[:, 1].clip(min=0, max=label.shape[0] - 1)
    mask = np.zeros_like(label, dtype=np.float32)
    # mask_pts = np.zeros_like(label, dtype=np.float32)
    cv2.fillPoly(mask, [pts_int], 1.0)
    # mask_pts = cv2.polylines(mask_pts, [pts_int], True, 1)
    # cv2.imwrite('mask.png', mask*255)
    # cv2.imwrite('label.png', label*255)
    # cv2.imwrite('pts.png', mask_pts*255)
    iou = iou_score(mask, label)
    return iou


def shape_to_mask(shape, WH):
    '''
    将shape变为二值化mask
    '''
    shape_int = np.int32(shape)
    shape_int[:, 0] = shape_int[:, 0].clip(min=0, max=WH[0] - 1)
    shape_int[:, 1] = shape_int[:, 1].clip(min=0, max=WH[1] - 1)
    # mask = np.zeros_like(label, dtype=np.float32)
    mask = np.zeros((WH[1], WH[0]))
    # mask_shape = np.zeros_like(label, dtype=np.float32)
    cv2.fillPoly(mask, [shape_int], 1.0)
    mask = mask.astype(np.uint8)
    return mask


def shape_to_boundary_set(shape, WH):
    '''
    将shape变为boundary的点集，方便计算hausdorff distance
    '''
    mask = shape_to_mask(shape, WH)
    boundary = cv2.Canny(mask, 0.3, 0.7)
    return boundary


def shape_to_iou(pts, label):
    '''
    计算每层outputs的最大值坐标，画封闭曲线，算iou
    pts: [49, 2] np.float32 resized_WH
    label: [H, W]
    '''
    mask = shape_to_mask(pts, (label.shape[1], label.shape[0]))
    iou = compute_iou(mask, label)
    return iou


# def wls_T(mean_shape, eig_vecs, pts, weights=1, b=0):
#     '''
#     用加权最小二乘回归参数，T和b只能同时回归一个
#     Params:
#         mean_shape: [49, 2]
#         eig_vecs: PCA modes, [5, 49, 2]
#         pts: target points, [49, 2]
#         weights: [49, 1]
#         b: PCA mode weight, [5,]，默认=0，每次迭代的初始值
#     Return:
#         pts_pos: 经WLS求参后，将shape_prior变换得到的后验点坐标，[49, 2]
#     '''
#     shape_vary = mean_shape + np.sum(eig_vecs * b, axis=0)  # [49, 2]
#     shape_vary_ones = sm.add_constant(shape_vary, False)  # [49, 3]
#     model = sm.WLS(pts, shape_vary_ones, weights)
#     results = model.fit()
#     pts_affined = model.predict(results.params)
#     return pts_affined, results.params

# def wls_b(mean_shape, eig_vecs, pts, T=np.ones((3, 3), np.int32), weights=1):
#     '''
#     加权最小二乘回归b，因为b有范围约束
#     eig_vecs的数量可以变化
#     '''
#     def residual(b, mean_shape, D, pts, weights, T, res=True):
#         # 误差函数
#         Db = []
#         bs = []
#         for i in range(D.shape[0]):  # vec_num
#             Db.append(D[i] * b['b' + str(i)])
#             bs.append(b['b' + str(i)])
#         Db = np.sum(Db, 0)  # [pts_num, 2]
#         shape_vary = Db + mean_shape
#         shape_vary_ones = np.column_stack([shape_vary, np.ones(shape_vary.shape[0])])
#         shape_affine = shape_vary_ones @ T  # [4, 2]
#         if res:
#             res = shape_affine - pts
#             weighted_res = res * weights
#             return weighted_res  # + sum([b**2 for b in bs])
#         else:
#             return shape_affine, np.array(bs)

#     b = Parameters()
#     for i in range(eig_vecs.shape[0]):
#         b.add('b' + str(i), 0, True, -0.1, 0.1)
#     out = minimize(residual, b, method='leastsq', args=(mean_shape, eig_vecs, pts, weights, T))
#     shape_affine, bs = residual(out.params, mean_shape, eig_vecs, pts, weights, T, False)
#     return shape_affine, bs

# def optimize_Tb(mean_shape, )


def seg_to_edge_prob(seg, save_name, save_dir=None, sigmma=2, gamma=100):
    '''
    将分割概率图转为边缘高斯概率图
    1. seg用OTSU二值化
    2. Canny算子找边缘
    3. 做高斯概率图
    seg: [H, W] np
    return: [H, W] np
    '''
    if len(np.unique(seg)) > 1:
        threshold = filters.threshold_otsu(seg, nbins=256)
    else:
        threshold = 0.5
    seg = np.where(seg > threshold, np.uint8(1), np.uint8(0))
    seg_reversed = np.where(seg > 0, np.uint8(0), np.uint8(1))
    dist_trans_inner = cv2.distanceTransform(seg, cv2.DIST_L2, 5)
    dist_trans_outer = cv2.distanceTransform(seg_reversed, cv2.DIST_L2, 5)
    dist_trans = dist_trans_inner + dist_trans_outer  # 内外都是正的
    gaussian_map = gamma / (2 * math.pi * sigmma**2) * np.exp(-0.5 * dist_trans / (sigmma**2))
    if save_dir is not None:
        factor = 255 / np.max(gaussian_map)
        cv2.imwrite(str(save_dir.joinpath(save_name)), gaussian_map * factor)
    return gaussian_map


def vis_pts_polylines_pre(pts, image, label, title=None, save_path=None):
    '''
    将soft_argmax得到的点可视化，画多边形，并保存
    pts: [49/4, 2] WH
    image: [H, W] np
    label: [H, W] np
    title: 图上写的字
    save_path: str，保存名字
    '''
    # image = image.squeeze()  # [H, W]
    # image = image/np.max(image)
    # label = label.squeeze()  # [H, W]
    label = cv2.resize(label, (image.shape[1], image.shape[0]), cv2.INTER_NEAREST)
    mask = np.zeros_like(image, dtype=np.float32)
    pts_int = np.int32(pts)
    pts_int[:, 0] = pts_int[:, 0].clip(min=0, max=image.shape[1] - 1)
    pts_int[:, 1] = pts_int[:, 1].clip(min=0, max=image.shape[0] - 1)
    mask = cv2.polylines(mask, [pts_int], isClosed=True, color=1, thickness=2)
    mask = mask + image * 0.15 + label * 0.15
    if title is not None:
        mask = cv2.putText(mask, title, (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1.0, 1.0, 2)
    if save_path is not None:
        cv2.imwrite(str(save_path), mask * 255)
    else:
        return torch.from_numpy(mask)


def vis_pts_polylines(pts, WH, image=None, label=None, pred=None, title=None, save_path=None, landmark=None):
    '''
    将soft_argmax得到的点可视化，画多边形，并保存
    pts: [49/4, 2] WH
    image: [H, W] or [H, W, 3]
    label: [H, W] or [H, W, 3]
    pred: [H, W] np, unet pred edge
    title: 图上写的字
    save_path: str，保存名字
    landmark: list of int, optional
    '''
    if image is not None and image.shape[-1] == 3:
        channel = 3
        mask = np.zeros((WH[1], WH[0], 3), dtype=np.float32)
    else:
        channel = 1
        mask = np.zeros((WH[1], WH[0]), dtype=np.float32)
    # pts_int = pts
    pts_int = np.int32(pts)
    pts_int[:, 0] = pts_int[:, 0].clip(min=0, max=WH[0] - 1)
    pts_int[:, 1] = pts_int[:, 1].clip(min=0, max=WH[1] - 1)
    if channel == 3:
        cv2.polylines(mask, [pts_int], isClosed=True, color=(0, 0, 255), thickness=1)
        if landmark is not None:
            for pt_idx in landmark:
                cv2.circle(mask, pts_int[pt_idx], 3, (0, 0, 255), 3)
    else:
        cv2.polylines(mask, [pts_int], isClosed=True, color=1, thickness=1)
        if landmark is not None:
            for pt_idx in landmark:
                cv2.circle(mask, pts_int[pt_idx], 3, 1, 3)
    cv2.line(mask, (10, 500), (50, 500), (0, 0, 255))
    cv2.putText(mask, 'ASM', (60, 500), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 1)
    if save_path is not None:
        if label is not None:
            label_edge = cv2.Canny(label, 30, 100)
            label_edge = cv2.cvtColor(label_edge, cv2.COLOR_GRAY2BGR)
            label_edge = np.where(label_edge > 0, (255, 255, 255), (0, 0, 0))  # 白色
            mask = mask + label_edge
            cv2.line(mask, (10, 550), (50, 550), (255, 255, 255))
            cv2.putText(mask, 'label', (60, 550), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
        if pred is not None:
            pred_edge = cv2.Canny(pred, 30, 100)
            pred_edge = cv2.cvtColor(pred_edge, cv2.COLOR_GRAY2BGR)
            pred_edge = np.where(pred_edge > 0, (0, 255, 0), (0, 0, 0))  # 绿色
            mask = mask + pred_edge
            cv2.line(mask, (10, 600), (50, 600), (0, 255, 0))
            cv2.putText(mask, 'Unet', (60, 600), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)
        if image is not None:
            mask = mask + image * 0.8
        if title is not None:
            # pos_x = WH[0] // 2 - 10 * len(title)
            pos_x = 30
            if channel == 3:
                cv2.putText(mask, title, (pos_x, 50), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255), 2)
            else:
                cv2.putText(mask, title, (pos_x, 50), cv2.FONT_HERSHEY_COMPLEX, 1.0, 1.0, 2)
        # scale = 255 / np.max(mask)
        scale = 1
        cv2.imwrite(str(save_path), mask * scale)
    else:
        return mask


def vis_shape(pts, WH, image, label, preds, save_path, pred_edge=False, title=None, landmark=None):
    '''
    产生论文用图，只画线，不写文字
    pts: [49/4, 2] WH
    image: [H, W]
    label: [H, W]
    preds: [H, W]x5 np, 5 unet pred
    save_path: str，保存名字
    pred_edge: bool
    '''
    temp = deepcopy(image)
    if pred_edge:  # 在image上画pred的边缘
        for pred in preds:
            pred = cv2.Canny(pred, 30, 100)
            pred_edge_contour = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            pred = cv2.drawContours(temp, pred_edge_contour, -1, (255, 0, 0), 2)  # 蓝色
    # 在image上画label的边缘
    label_edge = cv2.Canny(label, 30, 100)
    label_edge_contour = cv2.findContours(label_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    label_edge = cv2.drawContours(temp, label_edge_contour, -1, (0, 255, 0), 2)  # 绿色
    # 在image上画shape的边缘
    pts_int = np.int32(pts)
    pts_int[:, 0] = pts_int[:, 0].clip(min=0, max=WH[0] - 1)
    pts_int[:, 1] = pts_int[:, 1].clip(min=0, max=WH[1] - 1)
    cv2.drawContours(temp, [pts_int], -1, (0, 0, 255), 2)  # 红色
    if title is not None:
        pos_x = 30
        cv2.putText(temp, title, (pos_x, 50), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255), 2)
    if landmark is not None:
        for pt_idx in landmark:
            cv2.circle(temp, pts_int[pt_idx], 3, (0, 0, 255), 3)
    cv2.imwrite(str(save_path), temp)


def gaussian(pt, H, W, sigmma=1, gamma=100):
    '''
    根据pt位置，计算HW的mask各点高斯值，其中sigmma可以设定为可学习的参数。
    '''
    pt_W, pt_H = pt
    mask_W = np.matlib.repmat(pt_W, H, W)
    mask_H = np.matlib.repmat(pt_H, H, W)
    map_x = np.matlib.repmat(np.arange(W), H, 1)
    map_y = np.matlib.repmat(np.arange(H), W, 1)
    map_y = np.transpose(map_y)

    dist = np.sqrt((map_x - mask_W) ** 2 + (map_y - mask_H) ** 2)
    gaussian_map = gamma / (2 * math.pi * sigmma**2) * np.exp(-0.5 * dist / (sigmma**2))
    # cv2.imwrite('gaussian.png', gaussian_map*255)
    return gaussian_map


def vis_shape_prob(shape, image, label, sigmma=2, gamma=100, title=None, save_path=None):
    '''
    在pts位置构建高斯概率图
    pts: [49/4, 2] HW
    reutn: mask*0.7+image*0.15+label*0.15
    '''
    shape = shape.numpy()  # [4, 2]
    image = image.squeeze().numpy()
    image /= np.max(image)
    label = label.squeeze().numpy()  # [H, W]
    label = cv2.resize(label, (image.shape[1], image.shape[0]), cv2.INTER_NEAREST)
    mask = np.zeros_like(image, dtype=np.float32)
    for pt in shape:  # [2,]
        pt_mask = gaussian(pt, mask.shape[0], mask.shape[1], sigmma, gamma)
        mask += pt_mask
    mask /= np.max(mask)
    mask = mask * 0.7 + image * 0.15 + label * 0.15
    return torch.from_numpy(mask)


def pts_gaussian_pre(shape, H, W, sigmma=2, gamma=100):
    '''
    根据shape的坐标点，生成gaussian概率图
    shape: [49, 2] float32 WH
    '''
    shape_int = np.int32(shape)
    map_x = np.matlib.repmat(np.arange(W), H, 1)
    map_y = np.matlib.repmat(np.arange(H), W, 1)
    map_y = np.transpose(map_y)
    maps = []
    for pt in shape_int:
        pt_Ws = np.matlib.repmat(pt[0], H, W)
        pt_Hs = np.matlib.repmat(pt[1], H, W)
        dist = np.sqrt((map_x - pt_Ws) ** 2 + (map_y - pt_Hs) ** 2)
        gaussian_map = gamma / (2 * math.pi * sigmma**2) * np.exp(-0.5 * dist / (sigmma**2))
        maps.append(gaussian_map)
    return np.stack(maps, 0)


def pts_gaussian(shape, WH, sigmma=2, gamma=100):
    '''
    根据shape的坐标点，生成gaussian概率图
    shape: [49, 2] float32 WH
    '''
    W, H = WH
    shape_int = np.int32(shape)
    map_x = np.matlib.repmat(np.arange(W), H, 1)
    map_y = np.matlib.repmat(np.arange(H), W, 1)
    map_y = np.transpose(map_y)
    maps = []
    for pt in shape_int:
        pt_Ws = np.matlib.repmat(pt[0], H, W)
        pt_Hs = np.matlib.repmat(pt[1], H, W)
        dist = np.sqrt((map_x - pt_Ws) ** 2 + (map_y - pt_Hs) ** 2)
        gaussian_map = gamma / (2 * math.pi * sigmma**2) * np.exp(-0.5 * dist / (sigmma**2))
        maps.append(gaussian_map)
    return np.stack(maps, 0)


def shape_heatmap(shape, WH, sigmma=2, gamma=7):
    '''
    Step 1：shape变为mask
    Step 2：mask edge distance
    '''
    mask = shape_to_mask(shape, WH)
    mask_reversed = np.where(mask > 0, np.uint8(0), np.uint8(1))
    dist_trans_inner = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist_trans_outer = cv2.distanceTransform(mask_reversed, cv2.DIST_L2, 5)
    dist_trans = dist_trans_inner + dist_trans_outer  # 内外都是正的
    gaussian_map = gamma / (2 * math.pi * sigmma**2) * np.exp(-0.5 * dist_trans / (sigmma**2))
    # dist_trans = np.expand_dims(dist_trans, -1)
    # dist_trans = np.clip(dist_trans, -epsilon, +epsilon) / (epsilon*2)
    # gaussian_map_max = np.max(gaussian_map)
    # gaussian_map_max = 1 if gaussian_map_max == 0 else gaussian_map_max
    # factor = 255 / gaussian_map_max
    factor = 255 / np.max(gaussian_map)
    gaussian_map = gaussian_map * factor
    return gaussian_map


def vis_gaussian_maps(maps, image, label, title=None, save_path=None):
    '''
    将49个点的概率图可视化
    maps: [49, H, W] np
    image: [1, 1, H, W] tensor
    label: [1, 1, H, W] tensor
    '''
    avg_map = np.average(maps, axis=0)  # [H, W]
    avg_map /= np.max(avg_map)  # 归一化
    if title is not None:
        avg_map = cv2.putText(avg_map, title, (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1.0, 1.0, 2)
    if save_path is not None:
        cv2.imwrite(str(save_path), avg_map * 255)
    else:
        return torch.from_numpy(avg_map)


def make_video(H, W, path, fps=10):
    '''
    根据path里的frame做视频
    path: Path-class, 详细到单张样本
    '''
    frames = sorted([frame for frame in path.glob('*.png')])
    video = cv2.VideoWriter(str(path.joinpath(path.stem + '.avi')), cv2.VideoWriter_fourcc(*'MJPG'), fps, (W, H))
    for frame in frames:
        frame = cv2.imread(str(frame))
        video.write(frame)
    video.release()
