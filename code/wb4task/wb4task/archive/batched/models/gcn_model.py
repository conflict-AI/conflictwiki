
from torch import nn
from torch.nn import functional as F
from dgl import nn as dglnn

from wb0configs import configs
from wb4task.train_transductive.trans_batched_dataload_class import load_data
from wb4task.train_transductive.trans_batched_train_class import Trainer
from wb4task.models.pre_post_nns.edge_label_prediction import EdgeLabelPredictor
from wb4task.models.pre_post_nns.feature_reduction import FeatureReducer


class StochasticLayerGCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.conv1 = dglnn.GraphConv(in_features, hidden_features, weight=True, bias=True,
                                     allow_zero_in_degree=True, activation=nn.ReLU())
        self.conv2 = dglnn.GraphConv(hidden_features, out_features, weight=True, bias=True,
                                     allow_zero_in_degree=True, activation=nn.ReLU())
        self.conv3 = dglnn.GraphConv(hidden_features, out_features, weight=True, bias=True,
                                     allow_zero_in_degree=True, activation=nn.ReLU())

    def forward(self, blocks, x):
        x = F.relu(self.conv1(blocks[0], x))
        x = F.relu(self.conv2(blocks[1], x))
        x = F.relu(self.conv3(blocks[2], x))
        return x



class Model(nn.Module):
    def __init__(self, node_feature_dim, gnn_in_features, gnn_hidden_features, gnn_out_features):
        super().__init__()

        self.node_feature_reducer = FeatureReducer(node_feature_dim, gnn_in_features)
        self.node_feature_reducer = self.node_feature_reducer.apply(self.node_feature_reducer.init_weights) ## weight initialisation

        self.gnn = StochasticLayerGCN(gnn_in_features, gnn_hidden_features, gnn_out_features) ## message passing

        self.egde_label_predictor = EdgeLabelPredictor(gnn_out_features) ## edge label computation
        self.egde_label_predictor = self.egde_label_predictor.apply(self.egde_label_predictor.init_weights) ## weight initialisation


    def forward(self, edge_subgraph, blocks, x):
        # blocks is the graph
        # edge_subgraph is used for predictor only
        # x are node features
        h = self.node_feature_reducer(x)
        h = self.gnn(blocks, h)
        h = self.egde_label_predictor(edge_subgraph, h)
        return h




class Model_Trainer(Trainer):

    def __init__(self,wikinetworkdata, class_weights):
        super().__init__(wikinetworkdata, class_weights)


    def pass_data_to_model(self, model, input_nodes, edge_subgraph, blocks):

        # blocks = [b.to(torch.device('cuda')) for b in blocks]
        # edge_subgraph = edge_subgraph.to(torch.device('cuda'))
        # batch_size = edge_subgraph.num_edges()

        input_features = blocks[0].srcdata['node_features']  ## blocks message flow graphs are the neighborhood layers
        edge_predictions = model(edge_subgraph, blocks, input_features)
        edge_labels = edge_subgraph.edata['label_discrete'].float().view(-1, 1)

        return edge_predictions, edge_labels



if __name__ == "__main__":

    config = configs.ConfigBase()
    train_dl, val_dl, test_dl, label_info, wikinetworkdata = load_data(config, batch_size=128, n_val=0.1, n_test=0.1, neighborhood_steps=3)

    model = Model(node_feature_dim = 768, gnn_in_features=12, gnn_hidden_features=4, gnn_out_features=4)

    trainer = Model_Trainer(wikinetworkdata, label_info)
    model = trainer.train_model(train_dl, val_dl, model, n_epochs=20, early_stop=12, lr = 0.0001, weight_decay=1e-5)
    trainer.evaluate_model(model, test_dl)


