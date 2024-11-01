import logging
import numpy as np
from sklearn.datasets import make_moons
import torch

from constants import Constants

class Dataset:
    def __init__(self, dataset_name, args) -> None:
        self.dataset_name = dataset_name
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # TODO - Write functions to load different datasets
    
    def fetch_dataset(self):
        if self.dataset_name == "moons":
            return self.load_moons_with_linear_transformed_noise()
        else:
            raise Exception("Dataset not supported")
        
    def load_moons_with_linear_transformed_noise(self):
        logging.info(f"Loading moons dataset with linear transformed noise with {self.args['random_features']} random features")
        x_train, y_train =  make_moons(
            n_samples = self.args["moon_samples"], 
            noise = self.args["moon_noise"], 
            random_state= Constants.RANDOM_STATE
        )
        x_test, y_test =  make_moons(
            n_samples = int(self.args["moon_samples"] * Constants.TEST_SIZE), 
            noise = self.args["moon_noise"],
            random_state= Constants.RANDOM_STATE
        )

        random_transformation = np.random.randn(x_train.shape[1], self.args["random_features"])

        x_train_transformed = np.dot(x_train, random_transformation)
        x_test_transformed = np.dot(x_test, random_transformation)

        x_train = np.concatenate((x_train, x_train_transformed), axis = 1)
        x_test = np.concatenate((x_test, x_test_transformed), axis = 1)

        logging.info(f"x_train = {x_train.shape}, y_train = {y_train.shape}, x_test = {x_test.shape}, y_test = {y_test.shape}")

        x_train = x_train.astype(np.float32)
        y_train = y_train.astype(np.float32).reshape(-1, 1)
        x_test = x_test.astype(np.float32)
        y_test = y_test.astype(np.float32).reshape(-1, 1)

        x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0)
        x_test = (x_test - x_train.mean(axis=0)) / x_train.std(axis=0)

        x_train_tensor = torch.from_numpy(x_train).to(self.device)
        x_test_tensor = torch.from_numpy(x_test).to(self.device)

        y_train_tensor = torch.from_numpy(y_train).to(self.device)
        y_test_tensor = torch.from_numpy(y_test).to(self.device)

        if self.args["batching"]:
            train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor), batch_size=self.args["batch_size"], shuffle=True)
            test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor), batch_size=self.args["batch_size"], shuffle=True)            
        else:
            train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor), batch_size=len(x_train), shuffle=True)
            test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor), batch_size=len(x_test), shuffle=True)

        return train_loader, test_loader, {
            "x_train" : x_train,
            "x_test" : x_test,
            "y_train" : y_train,
            "y_test" : y_test
        } 

