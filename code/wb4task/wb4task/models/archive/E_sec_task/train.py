from wb0configs import configs
from wb4task.dataload_class import load_data
from wb4task.models.E_sec_task.model import Model

import torch
from torch import nn
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


class E_Sec_Trainer():

    def __init__(self, wikidataset, entity_emb = "e_sec"):
        self.wikidataset = wikidataset
        self.entity_emb = entity_emb

    def train_model(self, train_dl, val_dl, model, n_epochs=32, early_stop=3, lr = 0.0001, weight_decay=1e-5):
        loss_criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=lr, weight_decay=1e-5)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

        ## early stopping
        epochs_no_improve = 0
        min_val_loss = 999999

        ## epochs
        for epoch in range(n_epochs):

            ## training
            model.train()
            train_loss = 0
            train_acc = 0
            for i, (x_1, x_2, y, l) in enumerate(train_dl):

                optimizer.zero_grad()

                y_hat = model(x_1, x_2)
                loss = loss_criterion(y_hat, y)
                loss.backward()
                optimizer.step()
                # scheduler.step()
                train_loss += loss.item()

                y_hat = y_hat.clone().detach()
                y = y.clone().detach()
                y_hat_hot = (y_hat.clone().detach() > 0.5).float()
                train_acc += float((y_hat_hot == y).float().sum())

            ## validation
            model.eval()
            val_loss = 0
            val_acc = 0
            with torch.no_grad():  ## turn off gradients for validation
                for i, (x_1, x_2, y, l) in enumerate(val_dl):

                    y_hat = model(x_1, x_2)
                    loss = loss_criterion(y_hat, y)
                    val_loss += loss.item()

                    y = y.clone().detach()
                    y_hat = y_hat.clone().detach()
                    y_hat_hot = (y_hat > 0.5).float()  ## make 0 and 1
                    val_acc += float((y_hat_hot == y).float().sum())

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

            train_acc = round((train_acc) / len(train_dl.dataset.indices), 3)
            val_acc = round((val_acc) / len(val_dl.dataset.indices), 3)

            print(
                f'epoch: {epoch+1}/{n_epochs}, train loss: {round(train_loss,3)}, train acc: {train_acc}, val Loss: {round(val_loss,3)}, val acc: {val_acc}')

        return model


    def evaluate_model(self, model, test_dl):
        with torch.no_grad():  ## turn off gradients for validation

            X_1 = self.wikidataset.X_1[test_dl.dataset.indices]
            X_2 = self.wikidataset.X_2[test_dl.dataset.indices]
            Y = self.wikidataset.Y[test_dl.dataset.indices]
            #L = self.wikidataset.L[test_dl.dataset.indices]

            model.eval()

            ## function to implement
            y_hat = model(X_1, X_2)

            roc_auc = roc_auc_score(Y, y_hat.detach().numpy())
            y_hat_hot = (y_hat > 0.5).float().detach().numpy()
            Y = Y.detach().numpy()
            prec, rec, f1, support = precision_recall_fscore_support(Y, y_hat_hot, average=None)  # macro
            print("\nroc_auc:", roc_auc, "prec:", prec, "rec:", rec, "f1:", f1, "support:", support)


if __name__ == "__main__":

    config = configs.ConfigBase()
    train_dl, val_dl, test_dl, wikidataset = load_data(config, task_type= "entity_sec_task", batch_size=32, n_val=0.4, n_test=0.05, limit_samples = None)

    model = Model()
    model = model.apply(model.init_weights)

    trainer = E_Sec_Trainer(wikidataset, entity_emb = "e_sec")
    model = trainer.train_model(train_dl, val_dl, model, n_epochs=64, early_stop=12, lr = 0.0001, weight_decay=1e-5)
    trainer.evaluate_model(model, test_dl)
