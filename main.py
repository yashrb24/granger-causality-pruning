import json
import logging
import argparse
import torch
import torch.nn as nn

from dataset import Dataset
from model import neural_network
from constants import Constants
from utils import *
from feature_importances import FeatureImportance
from gc_prune import GCPruning
from lc_prune import LC_Pruning

parser = argparse.ArgumentParser(description='Causal Pruning')
parser.add_argument('--dataset', type=str, required=True, help='Dataset to be processed')
parser.add_argument('--moon_samples', type=int, default=None, help='Number of samples for moons dataset')
parser.add_argument('--moon_noise', type=float, default=None, help='Noise for moons dataset')
parser.add_argument('--random_features', type=int, default=None, help='Number of random features to be added to moons dataset')
parser.add_argument('--input_size', type=int, default=None, help='Input size for neural network')
parser.add_argument('--batching', action='store_true', help='Whether to use batching or not')
parser.add_argument('--batch_size', type=int, default=None, help='Batch size for batching')
parser.add_argument('--activation', type=str, default=None, help='Activation function to be used')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for training')
parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs for training')
parser.add_argument('--out_dir', type=str, default=None, help='Output directory for saving model and metrics')
parser.add_argument('--diff_order', type=int, default=0, help='Difference order for making time series stationary')
parser.add_argument('--gc_significance_alpha', type=float, default=0.01, help='Significance alpha for granger causality test')
parser.add_argument('--gc_n_lags', type=int, default=1, help='Number of lags for granger causality test')

def train_unpruned_model(model, trainloader, testloader, args, record_weights = True, save_freq = 100, data_np = None):
    model.train()
    if record_weights:
        recorded_weights = dict()
        recorded_weights["loss"] = []
        for name, _ in model.named_parameters():
            recorded_weights[name] = []

    optimizer = torch.optim.SGD(model.parameters(), lr=args["lr"], weight_decay = args["lr"])
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(args["num_epochs"]):
        for x,y in trainloader:
            x = x.to(args["device"])
            y = y.to(args["device"])
            y_pred = model(x)
            loss = loss_fn(y_pred, y.view(-1,1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if record_weights:
                for name, parameter in model.named_parameters():
                    value = parameter.cpu().detach().view(-1).numpy()
                    recorded_weights[name].append(value)

                recorded_weights["loss"].append(loss.item())

        if (epoch+1) % 100 == 0:
            logging.info(f"Epoch - {epoch + 1}: Loss - {loss.item()}")
            metrics = evaluate_performance(model, trainloader)
            logging.info(f"Train - Accuracy: {metrics['accuracy']}")#, AUC: {metrics['auc']}")#, Precision: {metrics['precision']}, Recall: {metrics['recall']}, F1: {metrics['f1']}")
        if (epoch+1) % save_freq == 0:
            save_dir_epoch = os.path.join(args["out_dir"], f"epoch_{epoch+1}")
            save_model_and_metrics(model, trainloader, testloader, save_dir_epoch, "unpruned_model.pt", "unpruned_model_performance.txt", recorded_weights)
            shap_values = FeatureImportance.fetch_shap_values(model, data_np["x_train"], data_np["x_test"])
            FeatureImportance.save_shap_values(shap_values, os.path.join(save_dir_epoch, "unpruned_model_shap_values.json"))    
    if record_weights:
        return model, recorded_weights
    else:
        return model, None

def main(args):
    seed_everything(Constants.RANDOM_STATE)

    # training unpruned model
    seed_everything(Constants.RANDOM_STATE)
    dataset = Dataset(args["dataset"], args)
    trainloader, testloader, data_np = dataset.fetch_dataset()
    model = neural_network([args["input_size"], 64, 64, 64, 64], 1, args["activation"], args["device"])
    model.to(args["device"])
    model, weights = train_unpruned_model(model, trainloader, testloader, args, record_weights = True, save_freq = 100, data_np = data_np)
    save_model_and_metrics(model, trainloader, testloader, args["out_dir"], "unpruned_model.pt","unpruned_model_performance.txt", weights)
    shap_values = FeatureImportance.fetch_shap_values(model, data_np["x_train"], data_np["x_test"])
    FeatureImportance.save_shap_values(shap_values, os.path.join(args["out_dir"], "unpruned_model_shap_values.json"))

    # perform gc_pruning
    gc_pruning = GCPruning(args)
    order = args["diff_order"]
    gc_alpha = args["gc_significance_alpha"]
    gc_n_lags = args["gc_n_lags"]

    model, perc_prune = gc_pruning.gc_prune(model, weights, args["out_dir"], order, gc_n_lags, gc_alpha)

    # evaluating gc_pruning
    save_model_and_metrics(model, trainloader, testloader, args["out_dir"], "gc_pruned_model.pt", "gc_pruned_model_performance.txt")
    shap_values = FeatureImportance.fetch_shap_values(model, data_np["x_train"], data_np["x_test"])
    FeatureImportance.save_shap_values(shap_values, os.path.join(args["out_dir"], "gc_pruned_model_shap_values.json"))
    logging.info(f"GC_Pruning complete, Percentage Pruned - {perc_prune}")

    # preparing for LC pruning
    logging.info("Performing LC_Pruning")
    seed_everything(Constants.RANDOM_STATE)
    lc_model = neural_network([args["input_size"], 64, 64, 64, 64], 1, args["activation"], args["device"])
    lc_model.to(args["device"])

    # performing LC pruning
    lc_pruning = LC_Pruning(args, trainloader, testloader)
    lc_model, perc_prune = lc_pruning.lc_prune(lc_model, perc_prune)
    logging.info(f"LC_Pruning complete, Percentage Pruned - {perc_prune}")

    # evaluating lc_pruning
    save_model_and_metrics(lc_model, trainloader, testloader, args["out_dir"], "lc_pruned_model.pt","lc_pruned_model_performance.txt")
    shap_values = FeatureImportance.fetch_shap_values(lc_model, data_np["x_train"], data_np["x_test"])
    FeatureImportance.save_shap_values(shap_values, os.path.join(args["out_dir"], "lc_pruned_model_shap_values.json"))

    # comparing feature importances
    gc_feature_discrepancy, gc_normalized_feature_discrepancy = compare_feature_importances(os.path.join(args["out_dir"], "unpruned_model_shap_values.json"), os.path.join(args["out_dir"], "gc_pruned_model_shap_values.json"))
    lc_feature_discrepancy, lc_normalized_feature_discrepancy = compare_feature_importances(os.path.join(args["out_dir"], "unpruned_model_shap_values.json"), os.path.join(args["out_dir"], "lc_pruned_model_shap_values.json"))

    with open(os.path.join(args["out_dir"], "feature_importance_discrepancy.txt"), "w") as f:
        f.write("GC Pruning:\n")
        f.write(f"Feature Discrepancy: {gc_feature_discrepancy}\n")
        f.write(f"Normalized Feature Discrepancy: {gc_normalized_feature_discrepancy}\n")
        f.write("LC Pruning:\n")
        f.write(f"Feature Discrepancy: {lc_feature_discrepancy}\n")
        f.write(f"Normalized Feature Discrepancy: {lc_normalized_feature_discrepancy}\n")



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    args = args.__dict__
    args["seed"] = Constants.RANDOM_STATE
    with open(os.path.join(args["out_dir"], "args.json"), "w") as f:
        json.dump(args, f, indent=4)
    args["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info(args)

    if args["dataset"] == "moons":
        assert args["moon_samples"] is not None
        assert args["moon_noise"] is not None
        assert args["random_features"] is not None
        assert args["input_size"] is not None and args["input_size"] == 2 + args["random_features"]
    
    if args["batching"]:
        assert args["batch_size"] is not None

    if not os.path.exists(args["out_dir"]):
        os.makedirs(args["out_dir"])

    main(args)
    