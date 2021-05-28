import torch
from torch import nn
from metrics import DiceScoreSoft

class DiceLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
 
    def forward(self, preds, true):        
        dice = DiceScoreSoft(preds, true, reduction=self.reduction)
        return 1 - dice
    
class DiceBCELoss(nn.Module):
    def __init__(self, w=0.5, reduction='mean'):
        super().__init__()
        self.w = w
        self.BCE = nn.BCELoss(reduction=reduction)
        self.Dice = DiceLoss(reduction=reduction)
 
    def forward(self, preds, true):        
        dice = self.Dice(preds, true)
        bce = self.BCE(preds, true)
        loss = self.w * bce + (1 - self.w) * dice
        return loss