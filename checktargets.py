import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from rbr.rbr import generate_recourse
import joblib
import numpy as np
import torch
import yaml
from sklearn.utils import check_random_state
from expt.common import clf_map, synthetic_params, train_func_map
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from utils import helpers
from utils.transformer import get_transformer
import classifiers
from expt.expt_config import ExptConfig
from expt.config import Config
from expt.expt_config import Expt5

import pandas as pd
import matplotlib.pyplot as plt

alphas3 = [i/100 for i in range(60,85,10)]
alphas3 += [i/100 for i in range(85,95,2)]
alphas3 += [i/100 for i in range(95,98,1)]
alphas3 += [i/1000 for i in range(980,1021,5)]
alphas3 += [i/100 for i in range(103,105,1)]
alphas3 += [i/100 for i in range(105,116,2)]
alphas3 += [i/100 for i in range(120,151,10)]

# 100*(np.array(alphas3)-1)
# alphas3 = alphas3[12:19]
alphas3 = [i/10000 for i in range(-11000,11001,1375)]



seed = 46
means = [0,0]
covs = [[1,0],[0,1]]
def clf(X0, X1):
    return X0 + X1 > 0
def get_clf(alpha):
    return lambda X0, X1: X0 + alpha*X1 > 0


def gen_data(means, covs, clf, points=1000):
    data1 = np.random.multivariate_normal(means, covs, points)
    data = pd.DataFrame(data1, columns=['X0', 'X1'])
    data['target'] = clf(data['X0'], data['X1'])
    data['target'] = data['target'].astype(float)
    y = data['target'].to_numpy()
    return data1, y

def train(model, X, y, lr=1e-3, num_epoch=100, verbose=False):
    model.train()
    criterion = nn.BCELoss(reduction="sum")
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.1)
    X = X.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    X = torch.tensor(X, dtype=torch.float32).cuda()
    y = torch.tensor(y, dtype=torch.float32).cuda()

    loss_diff = 1.0
    prev_loss = 0.0
    num_stable_iter = 0
    max_stable_iter = 3

    for i in range(num_epoch):
        X =X.cuda()
        y_pred = model(X)
        loss = criterion(y_pred.squeeze(), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if verbose:
            print("Iter %d: loss: %f" % (i, loss.data.item()))

        loss_diff = prev_loss - loss.data.item()

        if loss_diff <= 1e-7:
            num_stable_iter += 1
            if num_stable_iter >= max_stable_iter:
                break
        else:
            num_stable_iter = 0

        prev_loss = loss.data.item()

def runall(alphas3):
 
    inv = []
    for alpha in alphas3:

        print('data generated')
        x , y = gen_data(means, covs, clf)
        x = torch.from_numpy(x).cuda()
        y = torch.from_numpy(y).cuda()

        model = classifiers.mlp.Net0(2)
        model.cuda()



        train(model, x, y)        

        print('first model trained')

        x1, y1 = gen_data(means, covs, get_clf(alpha))
        x1 = torch.from_numpy(x1).cuda()
        y1 = torch.from_numpy(y1).cuda()

        model1 = classifiers.mlp.Net0(2)
        model1.cuda()

        train(model1, x1, y1)        
        print('both models trained')

        count = 0
        m  =[]
        for i in model.predict(x):
            if i == 0:
                m.append(x[count])
                count+=1


        print(len(m))
        ec = ExptConfig()
        new_config = ec.e5

        params = dict(
        train_data=x,
        config=new_config,
        device="cuda",
        perturb_radius=0.2,
        )

        cnt = 0
        cnt1 = 0
        for cf in m:
            random_state = check_random_state(seed)
            cnt1+=1
            if(model1.predict(generate_recourse(cf, model, random_state , params)) == 0):
                cnt+=1

        inv.append(cnt/cnt1)
    return inv    


inv = runall(alphas3)
t = -np.array(alphas3)
data1 = inv
print(inv)
print(alphas3)
# data2 = 100*np.array(M1accs3)+np.random.random(len(alphas3))
# data3 = 100*np.array(M2accs3)+np.random.random(len(alphas3))
# data2 = 100*np.array(M1accs3)
# data3 = 100*np.array(M2accs3)

plt.figure(dpi=180)
fig, ax1 = plt.subplots()
fig.set_dpi(150)

color = 'C1'
ax1.set_xlabel(r'$\alpha$ Shifting targets', fontsize=15)
ax1.set_ylabel('Invalidation (%)', fontsize=17)#, color='C1')
ax1.plot(t, data1, color=color, label='Invalidation of CF1 by M2', marker='o')
ax1.tick_params(axis='y')#, labelcolor='C1')
ax1.set_ylim(-1, 104)#14+max(data1))
ax1.axvline(0, linestyle=':')

ax1.legend(fontsize=14)

plt.show()
plt.savefig('outputtargets.png')
