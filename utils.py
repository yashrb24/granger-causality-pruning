import os
import json
import numpy as np
import torch
import random
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def evaluate_performance(net, test_loader):
    net.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for x, y in test_loader:
            x.to(net.device)
            y_pred.append(net.infer(x))
            y_true.append(y)
    y_pred = torch.cat(y_pred).cpu().detach().numpy()
    y_true = torch.cat(y_true).cpu().detach().numpy()

    y_pred = np.where(y_pred > 0.5, 1, 0)
    accuracy = accuracy_score(y_true, y_pred)
    results = {
        "accuracy": accuracy,
    }
    return results

def save_model_and_metrics(model, trainloader, testloader, out_dir, model_name, filename, weights = None):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    torch.save(model.state_dict(), os.path.join(out_dir, model_name))
    if weights is not None:
        torch.save(weights, os.path.join(out_dir, "weights_time_series.pt"))
    train_metrics = evaluate_performance(model, trainloader)
    test_metrics = evaluate_performance(model, testloader)
    with open(os.path.join(out_dir, filename), "w") as f:
        f.write("Train Metrics:\n")
        f.write(str(train_metrics))
        f.write("\n")
        f.write("Test Metrics:\n")
        f.write(str(test_metrics))
        f.write("\n")

def compare_feature_importances(file1, file2):
  with open(file1, 'r') as f:
    imp1 = json.load(f)
  with open(file2, 'r') as f:
    imp2 = json.load(f)

  sort1 = sorted(imp1.values(), reverse = True)
  sum1 = sum(sort1)
  sort2 = sorted(imp2.values(), reverse = True)
  sum2 = sum(sort2)

  total_diff = 0

  assert len(imp1) == len(imp2)
  for (key1, val1), (key2, val2) in zip(imp1.items(), imp2.items()):
    rank1 = sort1.index(val1)
    rank2 = sort2.index(val2)
    diff = abs(rank1 - rank2)
    diff_prop = abs(val1/sum1 - val2/sum2)

    print(key1, key2, diff, diff_prop)

    total_diff += diff * diff_prop

  return total_diff, total_diff/len(sort1)