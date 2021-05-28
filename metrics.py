import torch

def Accuracy(preds, true, threshold=0.5, reduction='mean'):
    preds = (preds > threshold).long()
    acc = (preds == true).float().mean(dim=(-1,-2))
    if reduction == 'mean':
        acc = acc.mean()
    elif reduction == 'sum':
        acc = acc.sum()
    return acc

def DiceScore(preds, true, threshold=0.5, eps=1e-5, reduction='mean'):
    preds = (preds > threshold).long()
    dice = 2 * (preds * true).sum(dim=(-1,-2)) + eps
    dice /= preds.sum(dim=(-1,-2)) + true.sum(dim=(-1,-2)) + eps
    if reduction == 'mean':
        dice = dice.mean()
    elif reduction == 'sum':
        dice = dice.sum()
    return dice

def IoU(preds, true, threshold=0.5, eps=1e-5, reduction='mean'):
    preds = (preds > threshold).long()
    iou = (preds * true).sum(dim=(-1,-2)) + eps
    iou /= ((preds + true) > 0).float().sum(dim=(-1,-2)) + eps
    if reduction == 'mean':
        iou = iou.mean()
    elif reduction == 'sum':
        iou = iou.sum()
    return iou

def DiceScoreSoft(preds, true, eps=1e-5, reduction='mean'):
    dice = 2 * (preds * true).sum(dim=(-1,-2)) + eps
    dice /= (preds ** 2).sum(dim=(-1,-2)) + (true ** 2).sum(dim=(-1,-2)) + eps
#     dice /= preds.sum(dim=(-1,-2)) + true.sum(dim=(-1,-2)) + eps
    if reduction == 'mean':
        dice = dice.mean()
    elif reduction == 'sum':
        dice = dice.sum()
    return dice

def IoUSoft(preds, true, eps=1e-5, reduction='mean'):
    intersection = (preds * true).sum(dim=(-1,-2))
    union = (preds ** 2).sum(dim=(-1,-2)) + (true ** 2).sum(dim=(-1,-2)) - intersection + eps
    intersection += eps
    iou = intersection / union
    if reduction == 'mean':
        iou = iou.mean()
    elif reduction == 'sum':
        iou = iou.sum()
    return iou