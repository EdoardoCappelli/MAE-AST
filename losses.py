import torch
import torch.nn.functional as F 

def mae_loss(target, recon_logits):
    # masked_target = target[bool_mask]
    # masked_recon = recon_logits[bool_mask]
    loss_recon = F.mse_loss(target, recon_logits)
    return loss_recon

def infoNCE_loss(target, class_logits):
    all_dots = torch.matmul(target, class_logits.transpose(-1, -2))
    log_softmax = torch.log_softmax(all_dots, dim=-1)
    loss_info_nce = -torch.mean(torch.diagonal(log_softmax, dim1=-2, dim2=-1))
    return loss_info_nce
