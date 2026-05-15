# model to run SBASC model.

from abc import ABC

from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import BertModel


class FocalLoss(nn.Module, ABC):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        """
        Initialize a Focal loss function introduced by Lin et al.
        https://doi.ieeecomputersociety.org/10.1109/TPAMI.2018.2858826
        :param gamma: the focussing parameter
        :param alpha: weighting factor
        :param size_average: whether to use averaging or sum
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class BERTLinear(nn.Module, ABC):
    def __init__(self, bert_type, num_cat, num_pol, gamma1=3, gamma2=3):
        """
        A torch model using BERT like neural-network structure for predicting joint sentiment and aspect category
        :param bert_type: BERT model to use can be from huggingface or path
        :param num_cat: total number of aspect categories
        :param num_pol: total number of polarities
        :param gamma1: focal loss gamma parameter for aspect loss
        :param gamma2: focal loss gamma parameter for sentiment loss
        """
        super().__init__()
        self.bert = BertModel.from_pretrained(
            bert_type, output_hidden_states=True)
        self.ff_cat = nn.Linear(768, num_cat)
        self.ff_pol = nn.Linear(768, num_pol)
        self.num_cat = num_cat
        self.num_pol = num_pol
        self.gamma1 = gamma1
        self.gamma2 = gamma2

    def forward(self, labels_cat, labels_pol, **kwargs):
        outputs = self.bert(**kwargs)
        x = outputs[2][11]  # (bsz, seq_len, 768)

        mask = kwargs['attention_mask']  # (bsz, seq_len)
        se = x * mask.unsqueeze(2)
        den = mask.sum(dim=1).unsqueeze(1)
        se = se.sum(dim=1) / den  # (bsz, 768)

        logits_cat = self.ff_cat(se)  # (bsz, num_cat)
        logits_pol = self.ff_pol(se)  # (bsz, num_pol)
        loss = FocalLoss(gamma=self.gamma1)(logits_cat, labels_cat) + FocalLoss(gamma=self.gamma2)(logits_pol,
                                                                                                   labels_pol)
        return loss, logits_cat, logits_pol
