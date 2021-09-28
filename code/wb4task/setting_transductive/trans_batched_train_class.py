
from abc import ABC, abstractmethod

import torch
torch.manual_seed(0)
from torch import nn
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve
import numpy as np
import statistics


class Trainer(ABC):

    def __init__(self, wikinetworkdata, label_info, task_setting):
        self.wikidataset = wikinetworkdata
        self.task_setting = task_setting
        self.class_weights = label_info["class_weights"]
        self.class_labels = label_info["class_labels"]


    def train_model(self, train_dl, val_dl, model, graph, n_epochs=32, early_stop=3, lr = 0.0001, weight_decay=1e-5):

        loss_criterion = nn.BCELoss()
        #loss_criterion = nn.NLLLoss(weight=class_weights)
        #optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


        ## early stopping
        epochs_no_improve = 0
        min_val_loss = 999999

        ## epochs
        for epoch in range(n_epochs):

            ## training
            model.train()
            train_loss = 0
            train_acc = []

            val_loss = 0
            val_acc = []

            for i, (input_nodes, edge_subgraph, blocks) in enumerate(train_dl):

                optimizer.zero_grad()
                y_hat, y = self.pass_data_to_model(model, "train", edge_subgraph, blocks)
                loss = loss_criterion(y_hat, y)
                #loss = self.weighted_binary_cross_entropy(y_hat, y, self.class_weights)

                loss.backward()
                optimizer.step()
                # scheduler.step()
                train_loss += loss.item()

                y_hat = y_hat.clone().detach()
                y = y.clone().detach()
                y_hat_hot = (y_hat > self.class_weights[np.where(self.class_labels == 1)[0][0]]).float()  ## make 0 and 1
                #y_hat_hot = (y_hat > 0.5).float()  ## make 0 and 1

                train_acc_batch = float((y_hat_hot == y).float().sum())
                train_acc.append(train_acc_batch / len(y))

            ## validation
            with torch.no_grad():  ## turn off gradients for validation

                for i, (input_nodes, edge_subgraph, blocks) in enumerate(val_dl):

                    y_val_hat, y_val = self.pass_data_to_model(model, "val", edge_subgraph, blocks)

                    model.eval()
                    optimizer.zero_grad()

                    loss = loss_criterion(y_val_hat, y_val)
                    #loss = self.weighted_binary_cross_entropy(y_hat, y, self.class_weights)
                    val_loss += loss.item()

                    y_val = y_val.clone().detach()
                    y_val_hat = y_val_hat.clone().detach()
                    #y_val_hat_hot = (y_val_hat > 0.5).float()  ## make 0 and 1
                    y_val_hat_hot = (y_val_hat > self.class_weights[np.where(self.class_labels == 1)[0][0]]).float()  ## make 0 and 1
                    val_acc_batch = float((y_val_hat_hot == y_val).float().sum())
                    val_acc.append(val_acc_batch / len(y_val))

            train_loss /= len(train_dl)
            val_loss /= len(val_dl)

            ## early stopping n val loss
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                epochs_no_improve = 1
            else:
                epochs_no_improve += 1
                if epochs_no_improve == early_stop + 1:
                    print("early stopping")
                    break

            #train_acc = round((train_acc) / float(graph.edata["train_mask"].sum()), 3)
            #val_acc = round((val_acc) / float(graph.edata["val_mask"].sum()), 3)
            train_acc = round(statistics.mean(train_acc), 3)
            val_acc = round(statistics.mean(val_acc), 3)

            print(f'epoch: {epoch+1}/{n_epochs}, train loss: {round(train_loss,3)}, train acc: {train_acc}, val Loss: {round(val_loss,3)}, val acc: {val_acc}')

        return model



    def evaluate_model(self, model, dl):
        with torch.no_grad():  ## turn off gradients for validation

            for i, (input_nodes, edge_subgraph, blocks) in enumerate(dl):

                ## function to implement
                y_hat, y = self.pass_data_to_model(model, "test", edge_subgraph, blocks)

            roc_auc = roc_auc_score(y.detach().numpy(), y_hat.detach().numpy())
            fpr, tpr, thresholds = roc_curve(y.detach().numpy(), y_hat.detach().numpy())
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]

            print("optimal_threshold:", optimal_threshold)
            y_hat_hot = (y_hat > optimal_threshold).float() ## make 0 and 1

            prec, rec, f1, support = precision_recall_fscore_support(y.detach().numpy(), y_hat_hot.detach().numpy(), average="binary")  # macro
            print("\nroc_auc:", roc_auc, "prec:", prec, "rec:", rec, "f1:", f1, "support:", support)



    @abstractmethod
    def pass_data_to_model(self,input_nodes, edge_subgraph, blocks):
        raise NotImplementedError("pass_data_to_model is not implemented")