
from torch import nn
from torch.nn import functional as F
from dgl import nn as dglnn

from wb0configs import configs
from wb4task.train_transductive.trans_dataload_class import load_data
from wb4task.train_transductive.trans_train_class import Trainer
from wb4task.models.pre_post_nns.edge_label_prediction import EdgeLabelPredictor
from wb4task.models.pre_post_nns.feature_reduction import FeatureReducer


class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(in_feats=in_feats, out_feats=hid_feats, aggregator_type='pool', norm=lambda x: F.normalize(x))
        self.conv2 = dglnn.SAGEConv(in_feats=hid_feats, out_feats=hid_feats, aggregator_type='mean')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = F.relu(self.conv1(graph, inputs))
        h = F.relu(self.conv2(graph, h))
        return h



class Model(nn.Module):
    def __init__(self, node_feature_dim, gnn_in_features, gnn_hidden_features, gnn_out_features):
        super().__init__()

        self.node_feature_reducer = FeatureReducer(node_feature_dim, gnn_in_features)
        self.node_feature_reducer = self.node_feature_reducer.apply(self.node_feature_reducer.init_weights) ## weight initialisation

        self.gnn = SAGE(gnn_in_features, gnn_hidden_features, gnn_out_features) ## message passing

        self.egde_label_predictor = EdgeLabelPredictor(gnn_out_features) ## edge label computation
        self.egde_label_predictor = self.egde_label_predictor.apply(self.egde_label_predictor.init_weights) ## weight initialisation


    def forward(self, g, x):
        # blocks is the graph
        # edge_subgraph is used for predictor only
        # x are node features
        h = self.node_feature_reducer(x)
        h = self.gnn(g, h)
        h = self.egde_label_predictor(g, h)
        return h



class Model_Trainer(Trainer):

    def __init__(self,wikinetworkdata, class_weights):
        super().__init__(wikinetworkdata, class_weights)


    def pass_data_to_model(self, model, train_step, graph):


        edge_mask = graph.edata[train_step + '_mask']
        node_features = graph.ndata["node_features"]

        edge_predictions = model(graph, node_features)
        edge_predictions = edge_predictions[edge_mask]

        edge_labels = graph.edata['label_discrete'].float()
        edge_labels = edge_labels[edge_mask].view(-1, 1)

        return edge_predictions, edge_labels



if __name__ == "__main__":

    config = configs.ConfigBase()
    graph, label_info, wikinetworkdata = load_data(config, n_val=0.05, n_test=0.05)

    model = Model(node_feature_dim = 768, gnn_in_features = 64, gnn_hidden_features = 64, gnn_out_features = 64)

    trainer = Model_Trainer(wikinetworkdata, label_info)
    model = trainer.train_model(model, graph, n_epochs=900, early_stop=50, lr = 0.0001, weight_decay=1e-8)
    trainer.evaluate_model(model, graph)


