import torch
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

def cmd(X, X_test, K=5):
    """
    central moment discrepancy (cmd)
    objective function for keras models (theano or tensorflow backend)
    
    - Zellinger, Werner, et al. "Robust unsupervised domain adaptation for
    neural networks via moment alignment.", TODO
    - Zellinger, Werner, et al. "Central moment discrepancy (CMD) for
    domain-invariant representation learning.", ICLR, 2017.
    """
    x1 = X
    x2 = X_test
    mx1 = x1.mean(0)
    mx2 = x2.mean(0)
    sx1 = x1 - mx1
    sx2 = x2 - mx2
    dm = l2diff(mx1,mx2)
    scms = [dm]
    for i in range(K-1):
        # moment diff of centralized samples
        scms.append(moment_diff(sx1,sx2,i+2))
        #scms+=moment_diff(sx1,sx2,1)
    return sum(scms).item()

def l2diff(x1, x2):
    """
    standard euclidean norm
    """
    return (x1-x2).norm(p=2)

def moment_diff(sx1, sx2, k):
    """
    difference between moments
    """
    ss1 = sx1.pow(k).mean(0)
    ss2 = sx2.pow(k).mean(0)
    #ss1 = sx1.mean(0)
    #ss2 = sx2.mean(0)
    return l2diff(ss1,ss2)


path = "/home/shivama2/pre-training-via-denoising/experiments/finetuning3/features/"
path_train = path+"train/"
path_test = path+"test/"
# res = []
sum_train = 0 

with open(path+"cmd", "rb") as fp:   # Unpickling
    b = pickle.load(fp)    
# for train_tensors in os.listdir(path_train):
#     train_features = torch.load(path_train+train_tensors).detach().cpu()
#     for test_tensors in os.listdir(path_test):
#         test_features = torch.load(path_test+test_tensors).detach().cpu()
#         res.append(cmd(train_features,test_features))

# with open(path+"cmd", "wb") as fp:   #Pickling
#     pickle.dump(res, fp)
# print(res)


path = "/home/shivama2/pre-training-via-denoising/experiments/iid_split/features/"
path_train = path+"train/"
path_test = path+"test/"

with open(path+"cmd", "rb") as fp:   # Unpickling
    a = pickle.load(fp)
# res2 = []
# sum_train = 0 
# for train_tensors in os.listdir(path_train):
#     train_features = torch.load(path_train+train_tensors).detach().cpu()
#     for test_tensors in os.listdir(path_test):
#         test_features = torch.load(path_test+test_tensors).detach().cpu()
#         res2.append(cmd(train_features,test_features))

# with open(path+"cmd", "wb") as fp:   #Pickling
    # pickle.dump(res2, fp)

# print(res2)


plt.boxplot((a,b),showfliers=False)
plt.show()
plt.savefig(path+'box_plot.png')

