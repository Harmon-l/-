import time
from tqdm import tqdm
import numpy as np
import cv2

import torch

from tools.common import ClassificationDataPrefetcher, AverageMeter, accuracy
from tools.data_augment import Cutmix, Cutout, Mixup

save_path = '/home/mindspore/HW/pj1/aug_image/'


def validate_classification(val_loader, model, criterion, config):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        model_on_cuda = next(model.parameters()).is_cuda
        for images, targets in tqdm(val_loader):
            if model_on_cuda:
                images, targets = images.cuda(), targets.cuda()

            data_time.update(time.time() - end)
            end = time.time()

            outputs = model(images)
            batch_time.update(time.time() - end)

            loss = criterion(outputs, targets)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            end = time.time()

    # per image data load time(ms) and inference time(ms)
    per_image_load_time = data_time.avg / config.batch_size * 1000
    per_image_inference_time = batch_time.avg / config.batch_size * 1000

    return top1.avg, top5.avg, losses.avg, per_image_load_time, per_image_inference_time


def train_classification(train_loader, model, criterion, optimizer, scheduler,
                         epoch, logger, config):
    '''
    train classification model for one epoch
    '''
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    local_rank = torch.distributed.get_rank() if config.distributed else None
    if config.distributed:
        gpus_num = torch.cuda.device_count()
        iters = len(train_loader.dataset) // (
            config.batch_size * gpus_num) if config.distributed else len(
                train_loader.dataset) // config.batch_size
    else:
        iters = len(train_loader.dataset) // config.batch_size

    prefetcher = ClassificationDataPrefetcher(train_loader)
    images, targets = prefetcher.next()
    iter_index = 1
    k = 1

    if config.data_augment == 'cutmix':
        data_augment = Cutmix(config.cutmix_beta)
    elif config.data_augment == 'cutout':
        data_augment = Cutout(config.cutout_lenght)
    elif config.data_augment == 'mixup':
        data_augment = Mixup(config.mixup_alpha)
    else:
        data_augment = None

    while images is not None:
        images, targets = images.cuda(), targets.cuda()
        if config.data_augment == 'cutmix':
            inputs, target_a, target_b, lam = data_augment(images, targets)
            # import pdb;pdb.set_trace()
            '''
            if k <= 3:
                save_img = np.asarray(inputs[0].permute(1,2,0).cpu())
                save_img = 1.0 / (1 + np.exp(-1 * save_img))
                save_img = np.round(save_img * 255)
                cv2.imwrite(save_path + 'cutmix_{}.png'.format(k), save_img)
                k += 1
            '''
            outputs = model(inputs)
            loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)
        elif config.data_augment == 'cutout':
            images = data_augment(images)
            '''
            if k <= 3:
                save_img = np.asarray(images[0].permute(1,2,0).cpu())
                save_img = 1.0 / (1 + np.exp(-1 * save_img))
                save_img = np.round(save_img * 255)
                cv2.imwrite(save_path + 'cutout_{}.png'.format(k), save_img)
                k += 1
            '''
            outputs = model(images)
            loss = criterion(outputs, targets)
        elif config.data_augment == 'mixup':
            inputs, target_a, target_b, lam = data_augment(images, targets)
            '''
            if k <= 3:
                save_img = np.asarray(inputs[0].permute(1,2,0).cpu())
                save_img = 1.0 / (1 + np.exp(-1 * save_img))
                save_img = np.round(save_img * 255)
                cv2.imwrite(save_path + 'mixup_{}.png'.format(k), save_img)
                k += 1
            '''
            outputs = model(inputs)
            loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
        
        loss = loss / config.accumulation_steps


        loss.backward()

        if iter_index % config.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # measure accuracy and record loss
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

        images, targets = prefetcher.next()

        if iter_index % config.print_interval == 0:
            log_info = f'train: epoch {epoch:0>4d}, iter [{iter_index:0>5d}, {iters:0>5d}], lr: {scheduler.get_lr()[0]:.6f}, top1: {acc1.item():.2f}%, top5: {acc5.item():.2f}%, loss: {loss.item():.4f}'
            logger.info(log_info) if (config.distributed and local_rank
                                      == 0) or not config.distributed else None

        iter_index += 1

    scheduler.step()

    return top1.avg, top5.avg, losses.avg
