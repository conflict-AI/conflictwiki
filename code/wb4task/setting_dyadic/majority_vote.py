from wb0configs import configs
from wb4task.setting_transductive.trans_batched_dataload_class import load_data
from wb4task.setting_transductive.trans_batched_train_class import Trainer
from wb4task.setting_dyadic.no_gnn import Model, Model_Trainer


import torch
torch.manual_seed(0)
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve
import numpy as np


def evaluate_model(dl):

    with torch.no_grad():  ## turn off gradients for validation

        for i, (input_nodes, edge_subgraph, blocks) in enumerate(dl):

            edge_labels = edge_subgraph.edata['label_discrete'].float()

            ## train / test
            edge_mask = edge_subgraph.edata['test_mask']
            edge_labels = edge_labels[edge_mask].view(-1, 1)
            y_hat = torch.ones(edge_labels.shape)

            ones = (edge_labels == 1.).sum(dim=0).item()
            total = (y_hat == 1.).sum(dim=0).item()
            #print(ones / total)

            roc_auc = roc_auc_score(edge_labels.detach().numpy(), y_hat.detach().numpy())
            fpr, tpr, thresholds = roc_curve(edge_labels.detach().numpy(), y_hat.detach().numpy())

            y_hat_hot = y_hat#(y_hat > optimal_threshold).float()  ## make 0 and 1
            prec, rec, f1, support = precision_recall_fscore_support(edge_labels.detach().numpy(), y_hat_hot.detach().numpy(),
                                                                     average="binary")  # macro
            print("\nroc_auc:", roc_auc, "prec:", prec, "rec:", rec, "f1:", f1, "support:", support)


def run_experiments():

    for data_seed in list(range(10)):

        dl, val_test_dl, graph, label_info, wikinetworkdata = load_data(config, batch_size = 512, n_val= 0.3, n_test= 0.1, neighborhood_steps= 1, random_node_frac = 0.0, random_edge_frac = 0.0, random_label_frac = 0.0, random_seed=data_seed)

        model = Model(node_feature_dim = 500, node_reducer_out = 512, dot_product_dim = 96, include_edge_features = True)
        evaluate_model(val_test_dl)




if __name__ == "__main__":

    config = configs.ConfigBase()
    run_experiments()
