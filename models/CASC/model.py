# model to run CASC model.
#
# Adapted from Kumar et. al. (2021). Changes have been made to adapt the methods to the
# proposed framework
# https://github.com/Raghu150999/UnsupervisedABSA
#
# Kumar, A., Gupta, P., Balan, R. et al. BERT Based Semi-Supervised Hybrid Approach for Aspect and Sentiment
# Classification. Neural Process Lett 53, 4207–4224 (2021). https://doi-org.eur.idm.oclc.org/10.1007/s11063-021-10596-6

import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel


class LQLoss(nn.Module):
    def __init__(self, cfg, q, weight, alpha=0.0):
        super().__init__()
        self.q = q ## parameter in the paper
        self.alpha = alpha ## hyper-parameter for trade-off between weighted and unweighted GCE Loss
        self.weight = nn.Parameter(F.softmax(torch.log(1. / torch.tensor(weight, dtype=torch.float32)), dim=-1), requires_grad=False).to(cfg.device) ## per-class weights

    def forward(self, input, target, *args, **kwargs):
        bsz, _ = input.size()

        Yq = torch.gather(input, 1, target.unsqueeze(1))
        lq = (1 - torch.pow(Yq, self.q)) / self.q

        _weight = self.weight.repeat(bsz).view(bsz, -1)
        _weight = torch.gather(_weight, 1, target.unsqueeze(1))
    
        return torch.square(torch.mean(self.alpha * lq + (1 - self.alpha) * lq * _weight))

class BERTLinear(nn.Module):
    def __init__(self, cfg, bert_type, num_cat, num_pol):
        super().__init__()
        self.cfg = cfg
        self.bert = BertModel.from_pretrained(
            bert_type, output_hidden_states=True)
        self.ff_cat = nn.Linear(768, num_cat)
        self.ff_pol = nn.Linear(768, num_pol)
        self.aspect_weights = cfg.domain.aspect_weights
        self.sentiment_weights = cfg.domain.sentiment_weights

    def forward(self, labels_cat, labels_pol, **kwargs):
        outputs = self.bert(**kwargs)
        x = outputs[2][11]  # (bsz, seq_len, 768)

        mask = kwargs['attention_mask']  # (bsz, seq_len)
        se = x * mask.unsqueeze(2)
        den = mask.sum(dim=1).unsqueeze(1)
        se = se.sum(dim=1) / den  # (bsz, 768)

        logits_cat = self.ff_cat(se)  # (bsz, num_cat)
        logits_pol = self.ff_pol(se)  # (bsz, num_pol)
        if self.cfg.ablation.name == 'WithoutFocalLoss':
            loss = LQLoss(self.cfg, 0.4, self.aspect_weights, self.cfg.ablation.alpha)(F.softmax(logits_cat, dim=-1), labels_cat)
            loss = loss + LQLoss(self.cfg, 0.4, self.sentiment_weights, self.cfg.ablation.alpha)(F.softmax(logits_pol, dim=-1), labels_pol)
        else:
            loss = LQLoss(self.cfg, 0.4, self.aspect_weights)(F.softmax(logits_cat, dim=-1), labels_cat)
            loss = loss + LQLoss(self.cfg, 0.4, self.sentiment_weights)(F.softmax(logits_pol, dim=-1), labels_pol)
        return loss, logits_cat, logits_pol

