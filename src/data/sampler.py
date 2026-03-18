from torch.utils.data.sampler import Sampler
import random
import numpy as np

class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, labels=None):
        self.labels = labels
        self.dataset = dict()
        self.balanced_max = 0
        
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            self.balanced_max = len(self.dataset[label]) \
                if len(self.dataset[label]) > self.balanced_max else self.balanced_max
        
        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))
        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1]*len(self.keys)

    def __iter__(self):
        while self.indices[self.currentkey] < self.balanced_max - 1:
            self.indices[self.currentkey] += 1
            yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
            self.currentkey = (self.currentkey + 1) % len(self.keys)
        self.indices = [-1]*len(self.keys)
   
    def _get_label(self, dataset, idx, labels = None):
        _, label = dataset[idx]
        return label
   
    def __len__(self):
        return self.balanced_max*len(self.keys)
   
class UnderSampler(Sampler):
    def __init__(self, dataset, labels=None, under_sample_rate=0.2):
        self.under_sample_rate = under_sample_rate
        self.dataset_full = dataset
        
        if labels is not None:
            self.labels = labels
        elif hasattr(dataset, 'get_labels') and callable(getattr(dataset, 'get_labels')):
            self.labels = dataset.get_labels()
        else:
            self.labels = [self._get_label(dataset, idx) for idx in range(len(dataset))]
        
        self.dataset = {}
        for idx, label in enumerate(self.labels):
            if label not in self.dataset:
                self.dataset[label] = []
            self.dataset[label].append(idx)
            
        self.under_represented_label = min(self.dataset, key=lambda x: len(self.dataset[x]))
        
        self.minority_size = len(self.dataset[self.under_represented_label])
        self.majority_sizes = {label: int(len(indices) * self.under_sample_rate) 
                            for label, indices in self.dataset.items() 
                            if label != self.under_represented_label}
        
        self._length = self.minority_size + sum(self.majority_sizes.values())
        
    def _get_label(self, dataset, idx, labels=None):
        _, label = dataset[idx]
        return label
    
    def __len__(self):
        return self._length
    
    def __iter__(self):
        minority_indices = np.array(self.dataset[self.under_represented_label])
        
        under_sampled_indices = list(minority_indices)
        
        for label, indices in self.dataset.items():
            if label != self.under_represented_label:
                indices = np.array(indices)
                sample_size = self.majority_sizes[label]
                
                if sample_size < len(indices):
                    sampled = indices[np.random.choice(len(indices), sample_size, replace=False)]
                    under_sampled_indices.extend(sampled)
                else:
                    under_sampled_indices.extend(indices)
        
        under_sampled_indices = np.array(under_sampled_indices)
        np.random.shuffle(under_sampled_indices)
        
        return iter(under_sampled_indices.tolist())
