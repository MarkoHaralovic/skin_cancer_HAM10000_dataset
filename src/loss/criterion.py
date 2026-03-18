from torch import long, nn
from torch.nn import functional as F
import torch
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

class OhemCrossEntropy(nn.Module):
   """Ohem cross entropy

   Args:
      ignore_label (int): ignore label
      thres (float): maximum probability of prediction to be ignored
      min_kept (int): maximum number of pixels to be consider to compute loss
      weight (torch.Tensor): weight for cross entropy loss
   """

   def __init__(self, ignore_label=-1, thres = 0.5,  weight=None):
      super(OhemCrossEntropy, self).__init__()
      self.top_k = thres
      self.ignore_label = ignore_label
      self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction="none")

   def forward(self, logits, targets):
      if self.ignore_label != -1:
            valid_mask = targets != self.ignore_label
            if not valid_mask.all():
                logits = logits[valid_mask]
                targets = targets[valid_mask]
      if logits.shape[0] == 0:
         return logits.sum()
      losses = self.criterion(logits, targets)
                
      if isinstance(self.top_k, float):
         k = int(self.top_k * losses.shape[0])  
      else:
         k = self.top_k
      _, indices = torch.topk(losses, k=k, largest = True)
      ohem_loss = losses[indices]
      
      return ohem_loss.mean()
   
class RecallCrossEntropy(nn.Module):
   def __init__(self, n_classes=7, weight=None, ignore_index=-1, eps=1e-2):
      super(RecallCrossEntropy, self).__init__()
      self.n_classes = n_classes
      self.criterion = nn.CrossEntropyLoss(reduction='none', weight=weight, ignore_index=ignore_index)
      self.ignore_index = ignore_index
      self.eps = eps
      
   def forward(self, logits, targets):
      pred = logits.argmax(dim=1)
      idex =  (pred != targets).view(-1) # get the index of the misclassified samples
      
      gt_counter = torch.ones((self.n_classes,), device=targets.device)
      gt_idx, gt_count = torch.unique(targets, return_counts=True)
      
      gt_counter[gt_idx] = gt_count.float()
      
      fn_counter = torch.ones((self.n_classes), device=targets.device)
      fn = targets.view(-1)[idex]
      fn_idx, fn_count = torch.unique(fn, return_counts=True)
      
      fn_counter[fn_idx] = fn_count.float()
      
      weight = fn_counter / gt_counter
      
      CE = self.criterion(logits, targets)
      
      loss =  weight[targets] * CE 
      
      return loss.mean() + self.eps
            
class FocalLoss(nn.Module):
   def __init__(self, gamma=0, alpha=None, size_average=True):
      super(FocalLoss, self).__init__()
      self.gamma = gamma
      self.alpha = alpha
      if isinstance(alpha, (float, int)): 
         self.alpha = torch.Tensor([alpha, 1-alpha])
      if isinstance(alpha, list): 
         self.alpha = torch.Tensor(alpha)
      self.size_average = size_average

   def forward(self, input, target):
      target = target.view(-1, 1)

      logpt = F.log_softmax(input, dim=1)
      logpt = logpt.gather(1, target)
      logpt = logpt.view(-1)
      pt = logpt.data.exp()

      if self.alpha is not None:
         if self.alpha.type() != input.data.type():
               self.alpha = self.alpha.to(input.device)
         at = self.alpha.gather(0, target.view(-1))
         logpt = logpt * at

      loss = -1 * (1-pt)**self.gamma * logpt
      if self.size_average: 
         return loss.mean()
      else: 
         return loss.sum()
        
def labels_to_class_weights(samples, num_classes=7):
   """
   Compute inverse-frequency class weights for the HAM10000 7-class taxonomy.

   Classes that appear rarely (e.g. df, vasc) receive higher weights so the
   loss function penalises mistakes on them more heavily.
   """
   labels = [sample[1] for sample in samples]

   class_counts = np.bincount(labels, minlength=num_classes).astype(float)
   logging.info(f"Class distribution: {class_counts}")

   class_counts[class_counts == 0] = 1

   weights = class_counts.sum() / (num_classes * class_counts)

   logging.info(f"Class weights: {weights}")
   return torch.tensor(weights, dtype=torch.float32)