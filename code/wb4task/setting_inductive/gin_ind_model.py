
from torch import nn
from torch.nn import functional as F
from dgl import nn as dglnn
import dgl
#from torch_geometric.nn.conv.signed_conv import SignedConv

from wb0configs import configs
from wb4task.setting_inductive.ind_batched_dataload_class import load_data
from wb4task.setting_inductive.ind_batched_train_class import Trainer

from wb4task.models.pre_post_nns.edge_label_prediction_cat import EdgeLabelPredictor
from wb4task.models.pre_post_nns.feature_reduction import FeatureReducer
from wb4task.models.gin_model import SignedGIN


class Model(nn.Module):
    def __init__(self, node_feature_dim, gnn_in_features, gnn_hidden_features, gnn_out_features, include_edge_features):
        super().__init__()

        self.node_feature_reducer = FeatureReducer(node_feature_dim, gnn_in_features)
        self.edge_feature_reducer = FeatureReducer(node_feature_dim, gnn_in_features)

        self.gnn = SignedGIN(gnn_in_features, gnn_hidden_features, gnn_out_features) ## message passing

        self.egde_label_predictor = EdgeLabelPredictor(gnn_out_features, include_edge_features= include_edge_features) ## edge label computation


    def forward(self, edge_subgraph, blocks):
        # blocks is the graph
        # edge_subgraph is used for predictor only
        # x are node features

        n = self.node_feature_reducer(blocks[0].srcdata['node_features']) ## input node features
        e = self.edge_feature_reducer(blocks[0].edata['edge_features']) ## input edge features # toDo: edge features incorproate

        n = self.gnn(edge_subgraph, blocks, n, e)

        e = self.edge_feature_reducer(edge_subgraph.edata['edge_features'])
        h = self.egde_label_predictor(edge_subgraph, n, e)
        return h




class Model_Trainer(Trainer):

    def __init__(self,wikinetworkdata, class_weights, task_setting):
        super().__init__(wikinetworkdata, class_weights, task_setting)


    def pass_data_to_model(self, model, edge_subgraph, blocks):

        # blocks = [b.to(torch.device('cuda')) for b in blocks]
        # edge_subgraph = edge_subgraph.to(torch.device('cuda'))
        # batch_size = edge_subgraph.num_edges()

        edge_predictions = model(edge_subgraph, blocks)
        edge_labels = edge_subgraph.edata['label_discrete'].float().view(-1, 1)

        return edge_predictions, edge_labels


if __name__ == "__main__":

    config = configs.ConfigBase()
    train_dl, val_dl, test_dl, label_info, wikinetworkdata  = load_data(config, batch_size = 512, n_val= 0.3, n_test= 0.1, neighborhood_steps= 2, random_node_frac = 0.0, random_label_frac = 0.0)

    model = Model(node_feature_dim = 500, gnn_in_features = 64, gnn_hidden_features = 64, gnn_out_features = 64, include_edge_features = True)

    trainer = Model_Trainer(wikinetworkdata, label_info, task_setting = "systemic")
    model = trainer.train_model(train_dl, val_dl, model, n_epochs=30, early_stop=12, lr = 0.001, weight_decay=1e-5)
    trainer.evaluate_model(model, test_dl)


