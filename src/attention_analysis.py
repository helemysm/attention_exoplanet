

import collections
import pickle

import matplotlib
import numpy as np
import seaborn as sns
import sklearn

from matplotlib import pyplot as plt
from matplotlib import cm
from sklearn import manifold
from box import box_from_file

import torch
from torch import nn


sns.set_style("whitegrid")


def data_iterator(data):
    for i, doc in enumerate(data):
        data_att = (doc[0]).squeeze(0).numpy()
        #print(data_att.shape)
        if i % 100 == 0 or i == len(data) - 1:
            print("{:.1f}% done".format(100.0 * (i + 1) / len(data)))
        yield [], data_att#
    

    
config = box_from_file(Path('config.yaml'), file_type='yaml')

data = torch.load(config.dataset.attdata_path)    
    
cont=0
uniform_attn_entropy = 0
entropies = np.zeros((8, 8))
entropies_cls = np.zeros((8, 8)) 


for tokens, attns in (data_iterator()):
    attns = 0.9999 * attns + (0.0001 / attns.shape[-1]) 
    
    uniform_attn_entropy -= np.log(1.0 / attns.shape[-1])
    
    entropies -= (attns * np.log(attns)).sum(-1).mean(-1)
    entropies_cls -= (attns * np.log(attns))[:, :, 0].sum(-1)
    cont = cont + 1

n_docs = len(data)
uniform_attn_entropy /= n_docs
entropies /= n_docs
entropies_cls /= n_docs



n_lay = 8
def get_data_points(head_data):
  xs, ys, avgs = [], [], []
  for layer in range(n_lay):
    for head in range(n_lay):
      ys.append(head_data[layer, head])
      xs.append(1 + layer)
    avgs.append(head_data[layer].mean())
  return xs, ys, avgs


#### 
xs, es, avg_es = get_data_points(entropies)
xs, es_cls, avg_es_cls = get_data_points(entropies_cls)

plt.figure(figsize=(12, 12))


def plot_entropies(ax, data, avgs, label, c):
    ax.scatter(xs, data, c=c, s=25, label=label)
    ax.plot(1 + np.arange(8), avgs)
    ax.plot([1, 8], [uniform_attn_entropy, uniform_attn_entropy],
          c="k", linestyle="--")
    ax.text(3, uniform_attn_entropy - 0.3, "uniform attention",
          ha="center")
    ax.legend(loc="lower right")
    ax.set_ylabel("Avg. Attention Entropy") #(nats)
    ax.set_xlabel("Layer")



plot_entropies(plt.subplot(2, 1, 1), es, avg_es, "Heads", c=BLUE)

plt.show()


    
    