from wb0configs import configs
from wb0configs.helpers import store_file, load_file, store_xml, load_xml

from torch.utils.data import Dataset, DataLoader, random_split
import torch
import numpy as np

class WikiDataset(Dataset):
    def __init__(self, config, limit_samples, task_type):

        self.config = config
        self.task_type = task_type

        data = load_file(self.config.get_path("task") / self.task_type / "data", ftype="pkl")
        labels = load_file(self.config.get_path("task") / self.task_type / "label", ftype="pkl")

        if limit_samples != None:
            data = (data_element[:limit_samples] for data_element in data)
            labels = labels[:limit_samples]

        if self.task_type == "conflict_entity_task":
            self.X_1, self.X_2, self.C, self.E, self.Y = data
            self.L = labels
        elif self.task_type == "entity_sec_task":
            self.X_1, self.X_2, self.Y = data
            self.L = labels

        self.get_class_weights()


    def __len__(self):
        return len(self.Y)


    def __getitem__(self, idx):
        if self.task_type == "conflict_entity_task":
            return [self.X_1[idx], self.X_2[idx], self.C[idx], self.E[idx], self.Y[idx], self.L[idx]]
        if self.task_type == "entity_sec_task":
            return [self.X_1[idx], self.X_2[idx], self.Y[idx], self.L[idx]]


    def get_splits(self, n_val=0.33, n_test=0.33):
        val_size = round(n_val * len(self.Y))
        test_size = round(n_test * len(self.Y))
        train_size = len(self.Y) - test_size - val_size
        return random_split(self, [train_size, val_size, test_size])


    def get_class_weights(self, ):
        self.target_labels, counts = np.unique(self.Y, return_counts=True)
        print("target labels:", self.target_labels, counts)
        self.target_weights = 1.0 / torch.tensor(counts, dtype=torch.float)


    def get_sampler(self, data):
        data_targets = data.dataset.Y[data.indices]
        sample_weights = torch.Tensor([self.target_weights[int(x[0])] for x in data_targets])
        sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(data_targets), replacement=True)
        return sampler



def load_data(config, task_type, batch_size=32, n_val=0.2, n_test=0.05, limit_samples = None):

    wikidataset = WikiDataset(config, limit_samples, task_type)
    train, val, test = wikidataset.get_splits(n_val=n_val, n_test=n_test)
    print("task_type:", task_type, ", batch size:", batch_size, ", train samples:", len(train), ", val samples:", len(val), ", test samples:", len(test))

    train_dl = DataLoader(train, batch_size=batch_size, sampler=wikidataset.get_sampler(train), shuffle=False)
    val_dl = DataLoader(val, batch_size=len(val), sampler=wikidataset.get_sampler(val), shuffle=False) ## single batch
    test_dl = DataLoader(test, batch_size=len(test), sampler=wikidataset.get_sampler(test), shuffle=False) ## single batch
    print("train_dl:", len(train_dl), "val_dl:", len(val_dl), "test_dl:", len(test_dl), "\n")
    return train_dl, val_dl, test_dl, wikidataset



if __name__ == "__main__":

    config = configs.ConfigBase()
    #train_dl, val_dl, test_dl, wikidataset = load_data(config, task_type = "conflict_entity_task", batch_size=32, n_val=0.2, n_test=0.05, limit_samples = None)
    train_dl, val_dl, test_dl, wikidataset = load_data(config, task_type = "entity_sec_task", batch_size=32, n_val=0.2, n_test=0.05, limit_samples = None)
    print(train_dl.dataset.indices)