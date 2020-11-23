import math

import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
import os

from config import device, fg_path_test, a_path_test, bg_path_test
from data_gen import data_transforms, fg_test_files, bg_test_files
from utils import compute_mse, compute_sad, AverageMeter, get_logger, compute_grad, compute_connectivity

def gen_test_names():
    num_fgs = 50
    num_bgs = 1000
    num_bgs_per_fg = 20

    names = []
    bcount = 0
    for fcount in range(num_fgs):
        for i in range(num_bgs_per_fg):
            names.append(str(fcount) + '_' + str(bcount) + '.png')
            bcount += 1

    return names


def process_test(im_name, bg_name):
    # print(bg_path_test + bg_name)
    im = cv.imread(fg_path_test + im_name)
    a = cv.imread(a_path_test + im_name, 0)
    h, w = im.shape[:2]
    bg = cv.imread(bg_path_test + bg_name)
    bh, bw = bg.shape[:2]
    wratio = w / bw
    hratio = h / bh
    ratio = wratio if wratio > hratio else hratio
    if ratio > 1:
        bg = cv.resize(src=bg, dsize=(math.ceil(bw * ratio), math.ceil(bh * ratio)), interpolation=cv.INTER_CUBIC)

    return composite4(im, bg, a, w, h)

def composite4(fg, bg, a, w, h):
    fg = np.array(fg, np.float32)

    bg_h, bg_w = bg.shape[:2]
    x = 0
    if bg_w > w:
        x = np.random.randint(0, bg_w - w)
    y = 0
    if bg_h > h:
        y = np.random.randint(0, bg_h - h)
    bg = np.array(bg[y:y + h, x:x + w], np.float32)
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[:, :, 0] = a / 255.
    im = alpha * fg + (1 - alpha) * bg
    im = im.astype(np.uint8)

    return im, bg, a

if __name__ == '__main__':
    save_root = 'images/KD'

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    checkpoint = 'BEST_checkpoint.tar'
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']

    model = model.to(device)
    model.eval()

    transformer = data_transforms['valid']

    names = gen_test_names()

    mse_losses = AverageMeter()
    sad_losses = AverageMeter()
    grad_losses = AverageMeter()
    connectivity_losses = AverageMeter()

    logger = get_logger()
    i = 0
    for name in tqdm(names):
        fcount = int(name.split('.')[0].split('_')[0])
        bcount = int(name.split('.')[0].split('_')[1])
        im_name = fg_test_files[fcount]
        bg_name = bg_test_files[bcount]
        trimap_name = im_name.split('.')[0] + '_' + str(i) + '.png'
        bg_name = bg_name.split('.')[0]
        fg_name = im_name.split('.')[0]

        img = cv.imread('data/merged_test/' + bg_name + '!' + fg_name + '!' + str(fcount) + '!' + str(bcount) +'.png')
        trimap = cv.imread('data/Combined_Dataset/Test_set/Adobe-licensed images/trimaps/' + trimap_name, 0)
        alpha = cv.imread(a_path_test + im_name, 0)

        i += 1
        if i == 20:
            i = 0

        #img, alpha, fg, bg, new_trimap = process_test(im_name, bg_name, trimap)
        #img, bg, alpha = process_test(im_name, bg_name)
        h, w = img.shape[:2]
        # save image
        # cv.imwrite('images/image.png', img)
        # mytrimap = gen_trimap(alpha)
        # cv.imwrite('images/test/new_im/'+trimap_name,mytrimap)

        x = torch.zeros((1, 4, h, w), dtype=torch.float)
        img = img[..., ::-1]  # RGB
        img = transforms.ToPILImage()(img)  # [3, 320, 320]
        img = transformer(img)  # [3, 320, 320]
        x[0:, 0:3, :, :] = img
        x[0:, 3, :, :] = torch.from_numpy(trimap.copy() / 255.)

        # Move to GPU, if available
        x = x.type(torch.FloatTensor).to(device)  # [1, 4, 320, 320]
        alpha = alpha / 255.

        with torch.no_grad():
            _, pred = model(x)  # [1, 4, 320, 320]

        pred = pred.cpu().numpy()
        pred = pred.reshape((h, w))  # [320, 320]

        pred[trimap == 0] = 0.0
        pred[trimap == 255] = 1.0
        cv.imwrite(os.path.join(save_root, trimap_name), pred * 255)

        mask = np.zeros([h, w])
        mask[trimap == 128] = 1
        w = np.sum(mask)
        # Calculate loss
        # loss = criterion(alpha_out, alpha_label)
        sad_loss = compute_sad(pred, alpha)
        mse_loss = compute_mse(pred, alpha, mask)
        grad_loss = compute_grad(pred, alpha, mask)
        connectivity_loss = compute_connectivity(pred, alpha, mask, step=0.1)

        str_msg = 'sad: %.4f, mse: %.4f, grad_loss: %.4f, con_loss: %.4f' % (
            sad_loss, mse_loss, grad_loss, connectivity_loss)

        print('test: {0}/{1}, '.format(i + 1, 20) + str_msg)

        sad_losses.update(sad_loss.item())
        mse_losses.update(mse_loss.item())
        grad_losses.update(grad_loss.item())
        connectivity_losses.update(connectivity_loss.item())
    print("SAD:{:0.2f}, MSE:{:0.4f}, GRAD:{:0.2f}, CON:{:0.2f}".format(sad_losses.avg, mse_losses.avg, grad_losses.avg,
                                                                     connectivity_losses.avg))
    with open(os.path.join(save_root + 'result.txt'),'a') as f:
        print("SAD:{:0.2f}, MSE:{:0.4f}, GRAD:{:0.2f}, CON:{:0.2f}".format(sad_losses.avg, mse_losses.avg, grad_losses.avg,
                                                                     connectivity_losses.avg), file=f)

