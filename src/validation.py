
from comet_ml import Experiment

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import math
import time
import tqdm
import pandas as pd

import logging

from sklearn import metrics

logger = logging.getLogger("exoclf")


def evaluate(self, loader: Dict[str, DataLoader], verbose: bool = False) -> None or float:

        running_loss = 0.0
        running_corrects = 0.0
        pbar = tqdm(total=len(loader.dataset))

        self.model.eval()
        self.experiment.log_parameter("test_ds_size", len(loader.dataset))

        atts_eval_list = []
        atts_eval_list_g = []
        atts_eval_list_st = []
        samples = []
        samples_global = []
        labels = []
        predicted_list = []
        samples_info = []
        steps_iter_test = 0
        acc_best = 0
        with self.experiment.test():
            with torch.no_grad():
                correct = 0.0
                total = 0.0
                #for x, y in enumerate(loader):
                cont_save = 0
                cont_num = 0
                outputs_list = []
                out_ebb_list_loc = []
                out_ebb_list_glob = []
                
                for x, x_cen, x_global, x_cen_global, x_stell, x_info, y in (loader):
                    
                    b_size = y.shape[0]
                    total += y.shape[0]
                    
                    x = x.to(self.device) if isinstance(x, torch.Tensor) else [i.to(self.device) for i in x]
                    x_cen = x_cen.to(self.device) if isinstance(x_cen, torch.Tensor) else [i.to(self.device) for i in x_cen]
                    
                    x_stell = x_stell.to(self.device) if isinstance(x_stell, torch.Tensor) else [i.to(self.device) for i in x_stell]
                    
                    x_info = x_info.to(self.device) if isinstance(x_info, torch.Tensor) else [i.to(self.device) for i in x_info]
                    
                    x_global = x_global.to(self.device) if isinstance(x_global, torch.Tensor) else [i.to(self.device) for i in x_global]
                    x_cen_global = x_cen_global.to(self.device) if isinstance(x_cen_global, torch.Tensor) else [i.to(self.device) for i in x_cen_global]
                    y = y.to(self.device)
                    
                    pbar.set_description("\033[32m"+"Evaluating"+"\033[0m")
                    pbar.update(b_size)

                    outputs,out_ebb_loc, out_ebb_glob, atts_eval, atts_eval_g, atts_eval_st = self.model(x, x_cen, x_global, x_cen_global, x_stell)
                    samples.append(x)
                    out_ebb_list_loc.append(out_ebb_loc)
                    out_ebb_list_glob.append(out_ebb_glob)
                    
                    samples_info.append(x_info)
                    
                    samples_global.append(x_global)
                    labels.append(y)
                    atts_eval_list.append(atts_eval.cpu().detach())
                    atts_eval_list_g.append(atts_eval_g.cpu().detach())
                    atts_eval_list_st.append(atts_eval_st.cpu().detach())
                    
                    loss = self.criterion(outputs, y)
                    _, predicted = torch.max(outputs, 1)
                    predicted_list.append(predicted)
                    outputs_list.append(outputs.to('cpu').numpy())
                    
                    correct += (predicted == y).sum().float().cpu().item()
    
                    running_loss += loss.cpu().item()
                    running_corrects += torch.sum(predicted == y).float().cpu().item()

                    self.experiment.log_metric("loss", running_loss, step=steps_iter_test)
                    self.experiment.log_metric("accuracy", float(running_corrects / total), step=steps_iter_test)
                    print('loss ', loss, 'loss ', loss.cpu().item(), ' accuracy ', float(running_corrects / total))
                            
                    steps_iter_test = steps_iter_test + 1
                 
                    
                pbar.close()
            acc = self.experiment.get_metric("accuracy")

        print("\033[33m" + "Evaluation finished. " + "\033[0m" + "Accuracy: {:.4f}".format(acc))

        #if verbose:
        #    return acc
        return atts_eval_list,atts_eval_list_g, atts_eval_list_st, samples, samples_global, labels, predicted_list, outputs_list,samples_info, out_ebb_list_loc, out_ebb_list_glob
        