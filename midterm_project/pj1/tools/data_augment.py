import torch
import torch.nn as nn
import numpy as np

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
 
 

 # generate mixed sample
class Cutmix(object):
    def __init__(self, beta = 1.0):
        self.beta = beta

    def __call__(self, input, target):
        lam = np.random.beta(self.beta, self.beta)
        rand_index = torch.randperm(input.size()[0]).cuda()
        target_a = target
        target_b = target[rand_index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
        input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
        
        return input, target_a, target_b, lam

        # output = model(input)
        # loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(2)
        w = img.size(3)

        mask = np.ones((h, w), np.float32)

        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)
        
        mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask).cuda()
        mask = mask.expand_as(img)
        img = img * mask

        return img

class Mixup(object):
    def __init__(self, alpha = 1.0):
        self.alpha = alpha

    def __call__(self, input, target):
        lam = np.random.beta(self.alpha, self.alpha)
        rand_index = torch.randperm(input.size()[0]).cuda()
        input = lam * input + (1 - lam) * input[rand_index, :]
        target_a, target_b = target, target[rand_index]

        return input, target_a, target_b, lam
        # output = model(input)
        # loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
