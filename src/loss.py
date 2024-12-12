### SPLICING PREDICTOR ###
# description:  loss functions
# author:       HPR

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# https://github.com/ashawkey/FocalLoss.pytorch/blob/master/Explaination.md
class FocalLossNLL2(nn.Module):
    def __init__(self, weight=None, gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        
        # alpha is replaced by loss weight
        # input = logpt
        pt = torch.exp(input)
        logpt = (1-pt)**self.gamma * input
        loss = F.nll_loss(logpt, target, self.weight)
        
        return loss


class SoftFBetaLoss(nn.Module):
    def __init__(self, beta = 1, weight=None, focal=False, alpha=0.25, gamma=2., epsilon=1e-06):
        nn.Module.__init__(self)

        self.focal = focal
        self.weight = weight
        self.beta = beta
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target):
        
        target = F.one_hot(target, num_classes=5)
        # ----- consider just the spliced class! -----
        target = target[:,4]
        input = input[:,4]
        # --------------------------------------------

        tp = target*input
        tn = (1-target)*(1-input)
        fp = (1-target)*input
        fn = target*(1-input)

        # in case target is 1
        prec = tp/(tp+fp+self.epsilon)
        rec = tp/(tp+fn+self.epsilon)

        # in case target is 0
        prec0 = tn/(tn+fn+self.epsilon)
        rec0 = tn/(tn+fp+self.epsilon)

        # calculate Fbeta scores
        fbeta = ((1+self.beta**2)*prec*rec)/((self.beta**2)*prec + rec + self.epsilon)
        fbeta0 = ((1+self.beta**2)*prec0*rec0)/((self.beta**2)*prec0 + rec0 + self.epsilon)

        # sum
        loss = target*(1-fbeta)+(1-target)*(1-fbeta0)

        if self.weight != None:
            loss = loss*self.weight

        if self.focal:
            loss = self.alpha*torch.pow((1-target), self.gamma)*loss

        return torch.mean(loss)


class CombinedLoss(nn.Module):
    def __init__(self, alpha1=1, alpha2=0.5, beta=1, weight=None, focal=False, gamma=2., epsilon=1e-06):
        nn.Module.__init__(self)

        self.focal = focal
        self.weight = weight
        self.beta = beta
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha1 = alpha1
        self.alpha2 = alpha2

        self.focal = FocalLossNLL2(weight=self.weight, gamma=self.gamma)
        self.fbeta = SoftFBetaLoss(beta=self.beta,focal=False)

    def forward(self, input, target):
        loss = self.alpha1*self.focal(input,target) + self.alpha2*self.fbeta(input,target)
        return loss



class LatentLoss(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, input, target):
        dp = nn.CosineSimilarity(dim=1)(input,target)
        return torch.mean(dp)



# ----- old/not working -----    
class FocalLossNLL(nn.Module):
    def __init__(self, weight=None, alpha=0.25, gamma=2., reduction='mean'):  # default gamma = 2.
        nn.Module.__init__(self)
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, input, target):
        nll_loss = nn.NLLLoss(weight=self.weight, reduction='none')
        loss = nll_loss(input, target)
        class_pred = input.gather(dim=1,index=target.unsqueeze(1)).squeeze(1)

        focal = self.alpha*torch.pow((1-class_pred), self.gamma)*loss
        
        if self.reduction == 'mean':
            return torch.mean(focal)
        elif self.reduction == 'sum':
            return torch.sum(focal)
        else:
            return focal


class FocalLossBCE(nn.Module):
    def __init__(self, weight=None, alpha=0.25, gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, input, target):
        bce_loss = F.binary_cross_entropy(input=input, target=target.float(),
                                          weight=self.weight, reduction=self.reduction)
        p_t = torch.exp(-bce_loss)
        F_loss = self.alpha * torch.pow((1-p_t),self.gamma)*bce_loss
        return F_loss


# NOTE: old
# use in latent space only!
class ReconstructionLoss(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, input, target):

        se = (input-target)**2
        return torch.mean(se)


class ScalableLoss(nn.Module):
    def __init__(self, weight=None, b=0.2, alpha=0.5, beta=0.1, delta=1., mode = 'P@R'):
        nn.Module.__init__(self)

        self.b = b  # threshold for positive class
        self.weight = weight  # loss weight for classes
        self.beta = beta  # recall
        self.alpha = alpha  # precision
        self.delta = delta  # > 0
        self.mode = mode
    
    def forward(self, input, target):
        target = F.one_hot(target, num_classes=5)
        input = torch.exp(input)
        
        plus = torch.where(input >= self.b)
        minus = torch.where(input < self.b)

        Yplus = len(plus[0])

        Lplus = F.binary_cross_entropy(input[plus[0],plus[1]].float(), target[plus[0],plus[1]].float())
        Lminus = F.binary_cross_entropy(input[minus[0],minus[1]].float(), target[minus[0],minus[1]].float())

        # maximise Fß
        if self.mode == 'Fbeta':
            phi = (Yplus-Lplus)
            l = (1/phi)*Lminus + self.delta*Lplus + ((1/phi)-self.delta)*Yplus + phi*self.delta
        # maximise P@Rß
        elif self.mode == 'P@R':
            l = Lminus + self.delta*(self.beta + (Lplus/Yplus) - 1)
        # maximise R@Pa
        elif self.mode == 'R@P':
            l = (1+self.delta)*Lplus + self.delta*(self.alpha/(1-self.alpha)*Lminus - self.delta*Yplus)

        
        if self.weight != None:
            l *= self.weight

        return l

# NOTE: does not work
class FbetaSurrogateLoss(nn.Module):
    def __init__(self, p, weight=None, beta=1, reduction='mean', average='weighted'):
        nn.Module.__init__(self)

        self.p = p
        self.weight = weight
        self.beta = beta
        self.average = average
        self.reduction = reduction # not used

    def forward(self, input, target):
        
        target = F.one_hot(target, num_classes=6)

        y0 = -1*target*torch.log(input)
        y1 = (1-target)*torch.log((self.beta**2)*(self.p/(1-self.p))+input)
        loss = (y0+y1)

        loss = torch.where(torch.isnan(loss), torch.ones_like(loss), loss)
        return torch.mean(loss*self.weight)