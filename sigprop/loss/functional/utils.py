import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def similarity_format(x,dim=1,no_std=False):
    ''' Calculate norm of x. '''
    if x.dim() > (dim+1):
        x = x.flatten(dim+1)
        if not no_std and x.size(dim+1) > 8:
            x = x.std(dim=dim+1)
        else:
            x = x.flatten(dim)
    xc = x - x.mean(dim=dim, keepdim=True)
    xn = xc / (1e-8 + torch.sqrt(torch.sum(xc**2, dim=dim, keepdim=True)))
    return xn

def similarity_matrix_x(x,dim=1):
    ''' Calculate normalized cosine similarity matrix of x[dim-1] x x[dim-1]. '''
    xn = similarity_format(x,dim)
    c = xn.matmul(xn.transpose(dim,dim-1)).clamp(-1,1)
    return c

def similarity_matrix_xy(x,y,dim=1):
    ''' Calculate normalized cosine similarity matrix of x[dim-1] x y[dim-1]. '''
    xn = similarity_format(x,dim)
    yn = similarity_format(y,dim)
    c = xn.matmul(yn.transpose(dim,dim-1)).clamp(-1,1)
    return c

def soft_target_cross_entropy(x, target, reduction='mean'):
    loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
    if reduction == 'mean':
        return loss.mean()
    else:
        return loss

def format_2dim_(h,format=True):
    h = h.flatten(1)
    if format:
        h = similarity_format(h, dim=1)
    return h

def format_4dim_(h,format=True):
    h = h.flatten(2).permute(2,0,1)
    if format:
        h = similarity_format(h, dim=2)
    return h

def sim_2dim_(h,t):
    yh = h @ t.t()
    return yh

def sim_4dim_(h,t):
    yh = (h @ t.permute(0,2,1)).mean(0)
    return yh

def format_sim_2dim_(h):
    h = format_2dim_(h)
    yh = sim_2dim_(h,h)
    return yh

def format_sim_4dim_(h):
    h = format_4dim_(h)
    yh = sim_4dim_(h,h)
    return yh
