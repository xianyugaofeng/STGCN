import torch
import torch.nn as nn

class MaskedMAELoss(nn.Module):
    # Masked Mean Absolute Error Loss
    def __init__(self, mask_value=0.0):
        super(MaskedMAELoss, self).__init__()
        self.mask_value = mask_value
    
    def forward(self, pred, target):
        # pred: (batch, output_length, num_nodes, num_features)
        # target: (batch, output_length, num_nodes, num_features)
        # Returns: scalar loss

        mask = (target != self.mask_value).float()
        loss = torch.abs(pred - target) * mask
        loss = loss.sum() / mask.sum() # 总误差除以有效位置的总数，得到平均绝对误差
        return loss

class MaskedMSELoss(nn.Module):
    # Masked Mean Squared Error Loss
    def __init__(self, mask_value=0.0):
        super(MaskedMSELoss, self).__init__()
        self.mask_value = mask_value
    
    def forward(self, pred, target):
        mask = (target != self.mask_value).float()
        loss = torch.square(pred - target) * mask
        loss = loss.sum() / mask.sum()
        return loss

class MaskedMAPELoss(nn.Module):
    # Masked Mean Absolute Percentage Error Loss
    def __init__(self, mask_value=0.0):
        super(MaskedMAPELoss, self).__init__()
        self.masked_value = masked_value
    
    def forward(self, pred, target):
        mask = (target != self.mask_value).float()
        loss = torch.abs((target - pred) / target) * mask * 100
        loss = loss.sum() / mask.sum()
        return loss

LOSS_ZOO = {
    'mae': nn.L1Loss(),
    'mse': nn.MSELoss(),
    'mape': nn.MAPELoss(),
    'masked_mae': MaskedMAELoss(),
    'masked_mse': MaskedMSELoss(),
    'masked_mape': MaskedMAPELoss()
}

def get_loss(loss_name):
    if loss_name not in LOSS_ZOO:
        raise ValueError(f"Loss {loss_name} not found in LOSS_ZOO")
    return LOSS_ZOO[loss_name]