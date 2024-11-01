import os
import logging
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests, pacf
from constants import Constants
import torch
from tqdm import tqdm

class GCPruning:
    def __init__(self, args) -> None:
        self.args = args
        self.order = args["diff_order"]

    def make_stationary(weights_dict, order):
        stationary_weights = dict()
        stationary_weights["loss"] = np.diff(np.asarray(weights_dict["loss"]), n = order).reshape(-1)
        for key in list(weights_dict.keys()):
            if key == "loss":
                continue
            stationary_weights[key] = dict()
            weights = np.array(weights_dict[key])
            for i in range(weights.shape[-1]):
                time_series = weights[:,i]
                time_series = np.diff(time_series, n=order)
                stationary_weights[key][i] = time_series
        return stationary_weights    
    
    def get_pacf_order(time_data, nlags=100):
        pacf_arr, confid_arr = pacf(time_data, nlags = nlags, alpha = 0.05)
        for i in range(nlags):
            if abs(pacf_arr[i]) < 0.05:
                return max(i - 1,1)
        return 1
    
    def check_granger_causality(weights_dict, significance_alpha, n_lags):
        gc_masks = dict()
        loss = weights_dict["loss"]

        for key in weights_dict.keys():
            if key == "loss":
                continue
            mask = []
            for index, time_series in tqdm(enumerate(weights_dict[key].values())):
                try:
                    data = np.concatenate([loss.reshape(-1,1), time_series.reshape(-1,1)], axis = 1)
                    gc_test = grangercausalitytests(data, [n_lags], verbose = False)
                    p_value = max([gc_test[n_lags][0][test][1] for test in Constants.GC_TESTS])
                    if p_value < significance_alpha:
                        mask.append(1)
                    else:
                        mask.append(0)
                except:
                    # logging.info(f"Error in GC test for {key} and index {index}")
                    mask.append(0)
            gc_masks[key] = mask
        return gc_masks
    
    def prune_model(model, masks):
        for name, parameter in model.named_parameters():
            mask = torch.Tensor(masks[name]).view(parameter.shape).to(parameter.device)
            parameter.data = torch.mul(mask, parameter.data)
        return model
    
    def gc_prune(self, model, weights, out_dir, order, gc_n_lags, gc_alpha):
        # if order != 0:
        weights = GCPruning.make_stationary(weights, order)
        masks = GCPruning.check_granger_causality(weights, gc_alpha, gc_n_lags)
        model = GCPruning.prune_model(model, masks)

        pruned = 0
        total = 0
        with open(out_dir + '/gc_pruning.txt', 'w') as f:
            for key in masks:
                count = len(masks[key]) - sum(masks[key])
                pruned += count
                total += len(masks[key])
                f.write(f'{key} layerwise pruning = {count/len(masks[key])}\n')
            f.write(f'Total pruning = {pruned/total}\n')

        return model, pruned/total
