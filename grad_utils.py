import torch
import pypose as pp
import mrob
import numpy as np

def mult(xi, x, eps = 1e-6):
    res = []
    for dxi in torch.eye(6)*eps:
        
        xi_ = pp.se3(xi + dxi)
        T = pp.Exp(xi_)
        p = T.Act(x)
        L0 = p@p.T

        xi_ = pp.se3(xi)
        T = pp.Exp(xi_)
        p = T.Act(x)
        L1 = p@p.T


        res.append(((L0-L1)/eps).tolist())
    return torch.Tensor(res).T

def mult_mrob(xi, x, eps = 1e-10):
    res = []
    for dxi in np.eye(6)*eps:
        
        xi_ = mrob.geometry.SE3(xi)
        dxi_ = mrob.geometry.SE3(dxi)
        p = (dxi_.mul(xi_)).transform(x)
        L0 = p @ p.T

        xi_ = mrob.geometry.SE3(xi)
        p = xi_.transform(x)
        L1 = p @ p.T

        res.append(((L0-L1)/(eps)).tolist())
    return torch.Tensor(res).T
