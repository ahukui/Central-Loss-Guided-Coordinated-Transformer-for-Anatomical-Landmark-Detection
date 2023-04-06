import torch
import torch.nn.functional as F
import torch.nn as nn

class Center_loss(nn.Module):
    def __init__(self, reduction: str = "sum"):
        super(Center_loss, self).__init__()
        self.reduction = reduction
    def forward(self,
                inputs: torch.Tensor,
                targets: torch.Tensor,
                gamma: float = 1):
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    
        Args:
            inputs (Tensor): A float tensor of arbitrary shape.
                    The predictions for each example.
            targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha (float): Weighting factor in range (0,1) to balance
                    positive vs negative examples or -1 for ignore. Default: ``0.25``.
            gamma (float): Exponent of the modulating factor (1 - p_t) to
                    balance easy vs hard examples. Default: ``2``.
            reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                    ``'none'``: No reduction will be applied to the output.
                    ``'mean'``: The output will be averaged.
                    ``'sum'``: The output will be summed. Default: ``'none'``.
        Returns:
            Loss tensor with the reduction option applied.
        """
    # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
        
        loss = -(targets*torch.abs((targets - inputs) ** gamma)*torch.log(inputs+1e-12) + 
                    (1-targets)*torch.abs((targets - inputs) ** gamma)*torch.log(1-inputs+1e-12))
    
        # Check reduction option and return loss accordingly
        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{self.reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
            )
        return loss


class focal_loss(nn.Module):
    def __init__(self, reduction: str = "sum"):
        super(focal_loss, self).__init__()
        self.reduction = reduction
    def forward(self,        
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2):
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    
        Args:
            inputs (Tensor): A float tensor of arbitrary shape.
                    The predictions for each example.
            targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha (float): Weighting factor in range (0,1) to balance
                    positive vs negative examples or -1 for ignore. Default: ``0.25``.
            gamma (float): Exponent of the modulating factor (1 - p_t) to
                    balance easy vs hard examples. Default: ``2``.
            reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                    ``'none'``: No reduction will be applied to the output.
                    ``'mean'``: The output will be averaged.
                    ``'sum'``: The output will be summed. Default: ``'none'``.
        Returns:
            Loss tensor with the reduction option applied.
        """
        # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
    

        p = inputs#torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)
    
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss
    
        # Check reduction option and return loss accordingly
        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{self.reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
            )
        return loss


class Weighted_CE(nn.Module):
    def __init__(self, reduction: str = "sum"):
        super(Weighted_CE, self).__init__()
        self.reduction = reduction
    def forward(self,
                inputs: torch.Tensor,
                targets: torch.Tensor):
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    
        Args:
            inputs (Tensor): A float tensor of arbitrary shape.
                    The predictions for each example.
            targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha (float): Weighting factor in range (0,1) to balance
                    positive vs negative examples or -1 for ignore. Default: ``0.25``.
            gamma (float): Exponent of the modulating factor (1 - p_t) to
                    balance easy vs hard examples. Default: ``2``.
            reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                    ``'none'``: No reduction will be applied to the output.
                    ``'mean'``: The output will be averaged.
                    ``'sum'``: The output will be summed. Default: ``'none'``.
        Returns:
            Loss tensor with the reduction option applied.
        """
    # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
        
        pos_num = torch.sum(targets)
        neg_num = torch.sum(1.- targets)
        loss = -(targets*(neg_num/(pos_num+neg_num))*torch.log(inputs+1e-12) + 
                    (1-targets)*(pos_num/(pos_num+neg_num))*torch.log(1-inputs+1e-12))
    
        # Check reduction option and return loss accordingly
        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{self.reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
            )
        return loss



# loss
l1 = torch.nn.L1Loss
l2 = torch.nn.MSELoss
bce = torch.nn.BCELoss
cl = Center_loss
fcl = focal_loss
wbce = Weighted_CE






# optimizer
adam = torch.optim.Adam
sgd = torch.optim.SGD
adagrad = torch.optim.Adagrad
rmsprop = torch.optim.RMSprop

# scheduler 
steplr = torch.optim.lr_scheduler.StepLR
multisteplr = torch.optim.lr_scheduler.MultiStepLR
cosineannealinglr = torch.optim.lr_scheduler.CosineAnnealingLR
reducelronplateau = torch.optim.lr_scheduler.ReduceLROnPlateau
lambdalr = torch.optim.lr_scheduler.LambdaLR

cycliclr = torch.optim.lr_scheduler.CyclicLR



