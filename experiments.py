# gc_significance_alphas  = [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1]
# orders = range(1,11)
# gc_n_lags = range(1,11)
# epochs = range(3000,10000,500)

import os
import torch
import logging

from dataset import Dataset
from model import neural_network
from utils import seed_everything, save_model_and_metrics
from main import train_unpruned_model
from feature_importances import FeatureImportance

logging.basicConfig(level=logging.INFO)
args = {
    "dataset": "moons",
    "moon_samples": 5000,
    "moon_noise": 0.5,
    "random_features": 8,
    "input_size": 10,
    "batching": True,
    "batch_size": 64,
    "activation": "relu",
    "lr": 0.001,
    "num_epochs": 150,
    "out_dir": "../output/",
    "diff_order": 0,
    "gc_significance_alpha": 0.01,
    "gc_n_lags": 1,
    "seed": 3
}

lrs = [0.0001, 0.0005, 0.001, 0.005, 0.01]

for lr in lrs:
    for seed in range(3):
        args["lr"] = lr
        out_dir = "../experiments/" + args["dataset"] + "_batching_l2reg_lr_" + str(lr) + "/seed_" + str(seed) + "/"
        args["out_dir"] = out_dir

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        seed_everything(seed)
        dataset = Dataset(args["dataset"], args)
        trainloader, testloader, data_np = dataset.fetch_dataset()
        model = neural_network([args["input_size"], 64, 64, 64, 64], 1, args["activation"], args["device"])
        model.to(args["device"])
        model, weights = train_unpruned_model(model, trainloader, testloader, args, record_weights = True, save_freq = 10, data_np = data_np)
        save_model_and_metrics(model, trainloader, testloader, args["out_dir"], "unpruned_model.pt","unpruned_model_performance.txt", weights)
        shap_values = FeatureImportance.fetch_shap_values(model, data_np["x_train"], data_np["x_test"])
        FeatureImportance.save_shap_values(shap_values, os.path.join(args["out_dir"], "unpruned_model_shap_values.json"))    