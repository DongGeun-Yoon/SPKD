import argparse
import logging
import os

import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm

import math

from config import im_size, epsilon, epsilon_sqr, device

from scipy.ndimage import gaussian_filter, morphology
from skimage.measure import label, regionprops

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(epoch, epochs_since_improvement, model, optimizer, loss, is_best):
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'loss': loss,
             'model': model,
             'optimizer': optimizer}
    # filename = 'checkpoint_' + str(epoch) + '_' + str(loss) + '.tar'
    filename = 'checkpoint.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_checkpoint.tar')


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def get_learning_rate(optimizer):
    return optimizer.param_groups[0]['lr']


def accuracy(scores, targets, k=1):
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def parse_args():
    parser = argparse.ArgumentParser(description='Train face network')
    # general
    parser.add_argument('--end-epoch', type=int, default=30, help='training epoch size.')
    parser.add_argument('--lr', type=float, default=0.01, help='start learning rate')
    parser.add_argument('--lr-step', type=int, default=10, help='period of learning rate decay')
    parser.add_argument('--optimizer', default='Adam', help='optimizer')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--mom', type=float, default=0.9, help='momentum')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size in each context')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint')
    parser.add_argument('--n_features', type=int, default=32, help='feature numbers')
    parser.add_argument('--KD_type', type=str, default='batch,spatial,channel', help='knowledge distillation type')
    parser.add_argument('--feature_layer', type=str, default='[1,2,3,4]', help='feature selected')
    parser.add_argument('--KD_weight', type=str, default='[1,1,1]', help='distillation loss weight')
    args = parser.parse_args()
    return args


def get_logger():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s \t%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


def safe_crop(mat, x, y, crop_size=(im_size, im_size)):
    crop_height, crop_width = crop_size
    if len(mat.shape) == 2:
        ret = np.zeros((crop_height, crop_width), np.uint8)
    else:
        ret = np.zeros((crop_height, crop_width, 3), np.uint8)
    crop = mat[y:y + crop_height, x:x + crop_width]
    h, w = crop.shape[:2]
    ret[0:h, 0:w] = crop
    if crop_size != (im_size, im_size):
        ret = cv.resize(ret, dsize=(im_size, im_size), interpolation=cv.INTER_NEAREST)
    return ret


# alpha prediction loss: the abosolute difference between the ground truth alpha values and the
# predicted alpha values at each pixel. However, due to the non-differentiable property of
# absolute values, we use the following loss function to approximate it.
def alpha_prediction_loss(y_pred, y_true, mask=None):
    if mask is not None:
        mask = mask
        #diff = y_pred[:, 0, :] - y_true
    else:
        mask = y_true[:, 1, :]
    diff = y_pred[:, 0, :] - y_true[:, 0, :]
    diff = diff * mask
    num_pixels = torch.sum(mask)

    return torch.sum(torch.sqrt(torch.pow(diff, 2) + epsilon_sqr)) / (num_pixels + epsilon)


# compute the MSE error given a prediction, a ground truth and a trimap.
# pred: the predicted alpha matte
# target: the ground truth alpha matte
# trimap: the given trimap
#
def compute_mse(pred, alpha, mask):
    num_pixels = mask.sum()
    return ((pred - alpha) ** 2).sum() / num_pixels


# compute the SAD error given a prediction and a ground truth.
#
def compute_sad(pred, alpha):
    diff = np.abs(pred - alpha)
    return np.sum(diff) / 1000


def compute_grad(pd, gt, mask):
    pd_x = gaussian_filter(pd, sigma=1.4, order=[1, 0], output=np.float32)
    pd_y = gaussian_filter(pd, sigma=1.4, order=[0, 1], output=np.float32)
    gt_x = gaussian_filter(gt, sigma=1.4, order=[1, 0], output=np.float32)
    gt_y = gaussian_filter(gt, sigma=1.4, order=[0, 1], output=np.float32)
    pd_mag = np.sqrt(pd_x ** 2 + pd_y ** 2)
    gt_mag = np.sqrt(gt_x ** 2 + gt_y ** 2)

    error_map = np.square(pd_mag - gt_mag)
    loss = np.sum(error_map * mask) / 10
    return loss



# compute the connectivity error
def compute_connectivity(pd, gt, mask, step=0.1):
    h, w = pd.shape

    thresh_steps = np.arange(0, 1.1, step)
    l_map = -1 * np.ones((h, w), dtype=np.float32)
    lambda_map = np.ones((h, w), dtype=np.float32)
    for i in range(1, thresh_steps.size):
        pd_th = pd >= thresh_steps[i]
        gt_th = gt >= thresh_steps[i]

        label_image = label(pd_th & gt_th, connectivity=1)
        cc = regionprops(label_image)
        size_vec = np.array([c.area for c in cc])
        if len(size_vec) == 0:
            continue
        max_id = np.argmax(size_vec)
        coords = cc[max_id].coords

        omega = np.zeros((h, w), dtype=np.float32)
        omega[coords[:, 0], coords[:, 1]] = 1

        flag = (l_map == -1) & (omega == 0)
        l_map[flag == 1] = thresh_steps[i - 1]

        dist_maps = morphology.distance_transform_edt(omega == 0)
        dist_maps = dist_maps / dist_maps.max()
        # lambda_map[flag == 1] = dist_maps.mean()
    l_map[l_map == -1] = 1

    # the definition of lambda is ambiguous
    d_pd = pd - l_map
    d_gt = gt - l_map
    # phi_pd = 1 - lambda_map * d_pd * (d_pd >= 0.15).astype(np.float32)
    # phi_gt = 1 - lambda_map * d_gt * (d_gt >= 0.15).astype(np.float32)
    phi_pd = 1 - d_pd * (d_pd >= 0.15).astype(np.float32)
    phi_gt = 1 - d_gt * (d_gt >= 0.15).astype(np.float32)
    loss = np.sum(np.abs(phi_pd - phi_gt) * mask) / 1000
    return loss

def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x + 1, y + 1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def over_all_loss(student_out, teacher_out, alpha, student_fms, teacher_fms,
                  KD_type, feature_maps, KD_weight):
    mask = alpha[:, 1, :]
    KD_weight = eval(KD_weight)
    l2 = nn.MSELoss()

    DS_loss = alpha_prediction_loss(student_out, alpha)
    TS_loss = alpha_prediction_loss(student_out, teacher_out, mask)

    loss = (DS_loss + TS_loss) / 2

    aggregated_student_fms = []
    aggregated_teacher_fms = []

    # using feature maps
    selected_student_fms = [student_fms[ind] for ind in eval(feature_maps)]
    selected_teacher_fms = [teacher_fms[ind] for ind in eval(feature_maps)]

    # for channel, FSP
    revised_student_fms = [student_fms[ind+9] for ind in eval(feature_maps)]
    revised_teacher_fms = [teacher_fms[ind] for ind in eval(feature_maps)]

    if 'hilton' not in KD_type:
        if 'batch' in KD_type:
            print('batch')
            aggregated_student_fms.append([batch_similarity(fm) for fm in selected_student_fms])
            aggregated_teacher_fms.append([batch_similarity(fm) for fm in selected_teacher_fms])
        if 'spatial' in KD_type:
            #print('S')
            aggregated_student_fms.append([spatial_similarity(fm) for fm in selected_student_fms])
            aggregated_teacher_fms.append([spatial_similarity(fm) for fm in selected_teacher_fms])
        if 'channel' in KD_type:
            #print('C')
            aggregated_student_fms.append([channel_similarity(fm) for fm in revised_student_fms])
            aggregated_teacher_fms.append([channel_similarity(fm) for fm in revised_teacher_fms])
        if 'FSP' in KD_type:
            print('F')
            aggregated_student_fms.append([FSP(revised_student_fms[i], revised_student_fms[i+1]) for i in range(len(revised_student_fms)-1)])
            aggregated_teacher_fms.append([FSP(revised_teacher_fms[i], revised_teacher_fms[i+1]) for i in range(len(revised_student_fms)-1)])
        if 'AT' in KD_type:
            print('AT')
            aggregated_student_fms.append([AT(fm) for fm in selected_student_fms])
            aggregated_teacher_fms.append([AT(fm) for fm in selected_teacher_fms])

    # KD loss
    for i in range(len(aggregated_student_fms)):
        for j in range(len(aggregated_student_fms[i])):
            loss += l2(aggregated_student_fms[i][j], aggregated_teacher_fms[i][j]) * KD_weight[i]

    return loss

class Distiller(nn.Module):
    def __init__(self, t_net, s_net):
        super(Distiller, self).__init__()

        teacher_bns = t_net.get_bn_before_relu()
        margins = [get_margin_from_BN(bn) for bn in teacher_bns]
        for i, margin in enumerate(margins):
            self.register_buffer('margin%d' % (i+1), margin.unsqueeze(1).unsqueeze(2).unsqueeze(0).detach())

        self.t_net = t_net
        self.s_net = s_net

    def forward(self, x):

        t_feats, t_out = self.t_net.extract_feature(x)
        s_feats, s_out = self.s_net.extract_feature(x)
        feat_num = 4

        loss_distill = 0
        for i in range(feat_num):
            loss_distill += distillation_loss(s_feats[i], t_feats[i].detach(), getattr(self, 'margin%d' % (i+1))) \
                            / 2 ** (feat_num - i - 1)

        return t_out, s_out, loss_distill

def get_margin_from_BN(bn):
    margin = []
    std = bn.weight.data
    mean = bn.bias.data
    for (s, m) in zip(std, mean):
        s = abs(s.item())
        m = m.item()
        if norm.cdf(-m / s) > 0.001:
            margin.append(- s * math.exp(- (m / s) ** 2 / 2) / math.sqrt(2 * math.pi) / norm.cdf(-m / s) + m)
        else:
            margin.append(-3 * s)

    return torch.FloatTensor(margin).to(std.device)

def distillation_loss(source, target, margin):
    target = torch.max(target, margin)
    loss = torch.nn.functional.mse_loss(source, target, reduction="none")
    loss = loss * ((source > target) | (target > 0)).float()
    return loss.sum()

def batch_similarity(fm): # batch similarity
    fm = fm.view(fm.size(0), -1)
    Q = torch.mm(fm, fm.transpose(0,1))
    normalized_Q = Q / torch.norm(Q,2,dim=1).unsqueeze(1).expand(Q.shape)
    return normalized_Q

def spatial_similarity(fm): # spatial similarity
    fm = fm.view(fm.size(0), fm.size(1),-1)
    norm_fm = fm / (torch.sqrt(torch.sum(torch.pow(fm,2), 1)).unsqueeze(1).expand(fm.shape) + 0.0000001 )
    s = norm_fm.transpose(1,2).bmm(norm_fm)
    s = s.unsqueeze(1)
    return s

def channel_similarity(fm): # channel_similarity
    fm = fm.view(fm.size(0), fm.size(1), -1)
    norm_fm = fm / (torch.sqrt(torch.sum(torch.pow(fm,2), 2)).unsqueeze(2).expand(fm.shape) + 0.0000001)
    s = norm_fm.bmm(norm_fm.transpose(1,2))
    s = s.unsqueeze(1)
    return s

def FSP(fm1, fm2):
    if fm1.size(2) > fm2.size(2):
        fm1 = F.adaptive_avg_pool2d(fm1, (fm2.size(2), fm2.size(3)))

    fm1 = fm1.view(fm1.size(0), fm1.size(1), -1)
    fm2 = fm2.view(fm2.size(0), fm2.size(1), -1).transpose(1,2)

    fsp = torch.bmm(fm1, fm2) / fm1.size(2)

    return fsp

def AT(fm):
    eps = 1e-6
    am = torch.pow(torch.abs(fm), 2)
    am = torch.sum(am, dim=1, keepdim=True)
    norm = torch.norm(am, dim=(2, 3), keepdim=True)
    am = torch.div(am, norm + eps)
    return am