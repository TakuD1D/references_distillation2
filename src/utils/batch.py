import torch

def batch_synthetic_index(syn_data, n):
    indices = torch.randperm(len(syn_data))
    batch_indices = list(torch.split(indices, n))
    return batch_indices[0]
def batch_synthetic_index_all(syn_data, n):
    indices = torch.randperm(len(syn_data))
    batch_indices = list(torch.split(indices, n))
    return batch_indices

def real_batch_index(real_data, n):
    indices = torch.randperm(len(real_data))
    batch_indices = list(torch.split(indices, n))
    return batch_indices