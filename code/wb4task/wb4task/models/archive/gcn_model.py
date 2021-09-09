
from torch import nn
from torch.nn import functional as F
from dgl import nn as dglnn
#from torch_geometric.nn.conv.signed_conv import SignedConv

from wb0configs import configs
from wb4task.setting_transductive.trans_batched_dataload_class import load_data
from wb4task.setting_transductive.trans_batched_train_class import Trainer
from wb4task.models.pre_post_nns.edge_label_prediction import EdgeLabelPredictor
from wb4task.models.pre_post_nns.feature_reduction import FeatureReducer


class SignedGCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()

        self.conv1 = dglnn.GraphConv(in_features, hidden_features)
        self.conv2 = dglnn.GraphConv(hidden_features, out_features)
        self.edge_norm = dglnn.EdgeWeightNorm(norm='both')


    def forward(self, blocks, x):
        #https://docs.dgl.ai/api/python/nn.pytorch.html?highlight=graphconv#dgl.nn.pytorch.conv.GraphConv

        edge_weight_0 = blocks[0].edata['label_discrete'].float() ## get edge signs
        edge_weight_0[edge_weight_0 == 0.0] = 0.0 ##edge weights need to be positive
        #norm_edge_weight_0 = self.edge_norm(blocks[0], edge_weight_0)
        norm_edge_weight_0 = edge_weight_0
        x = F.relu(self.conv1(blocks[0], x, edge_weight=norm_edge_weight_0))

        edge_weight_1 = blocks[1].edata['label_discrete'].float() ## get edge signs
        edge_weight_1[edge_weight_1 == 0.0] = 0.0  ##edge weights need to be positive
        #norm_edge_weight_1 = self.edge_norm(blocks[1], edge_weight_1)
        norm_edge_weight_1 = edge_weight_1
        x = F.relu(self.conv2(blocks[1], x, edge_weight=norm_edge_weight_1))
        return x



class Model(nn.Module):
    def __init__(self, node_feature_dim, gnn_in_features, gnn_hidden_features, gnn_out_features):
        super().__init__()

        self.node_feature_reducer = FeatureReducer(node_feature_dim, gnn_in_features)
        self.node_feature_reducer = self.node_feature_reducer.apply(self.node_feature_reducer.init_weights) ## weight initialisation

        self.gnn = SignedGCN(gnn_in_features, gnn_hidden_features, gnn_out_features) ## message passing

        self.egde_label_predictor = EdgeLabelPredictor(gnn_out_features) ## edge label computation
        self.egde_label_predictor = self.egde_label_predictor.apply(self.egde_label_predictor.init_weights) ## weight initialisation


    def forward(self, edge_subgraph, blocks, x):
        # blocks is the graph
        # edge_subgraph is used for predictor only
        # x are node features
        h = self.node_feature_reducer(x)
        h = self.gnn(blocks, h)
        h = self.egde_label_predictor(edge_subgraph, h, h)
        return h




class Model_Trainer(Trainer):

    def __init__(self,wikinetworkdata, class_weights):
        super().__init__(wikinetworkdata, class_weights)


    def pass_data_to_model(self, model, train_step, input_nodes, edge_subgraph, blocks):

        # blocks = [b.to(torch.device('cuda')) for b in blocks]
        # edge_subgraph = edge_subgraph.to(torch.device('cuda'))
        # batch_size = edge_subgraph.num_edges()

        input_features = blocks[0].srcdata['node_features']  ## blocks message flow graphs are the neighborhood layers
        edge_labels = edge_subgraph.edata['label_discrete'].float()
        edge_predictions = model(edge_subgraph, blocks, input_features)

        ## val
        val_edge_predictions = None
        val_edge_labels = None
        if train_step == "train":
            val_edge_mask = edge_subgraph.edata['val_mask']
            val_edge_predictions = edge_predictions[val_edge_mask]
            val_edge_labels = edge_labels[val_edge_mask].view(-1, 1)

        ## train / test
        edge_mask = edge_subgraph.edata[train_step + '_mask']
        edge_predictions = edge_predictions[edge_mask]
        edge_labels = edge_labels[edge_mask].view(-1, 1)

        return edge_predictions, edge_labels, val_edge_predictions, val_edge_labels


if __name__ == "__main__":

    config = configs.ConfigBase()
    dl, val_test_dl, graph, label_info, wikinetworkdata = load_data(config, batch_size = 32, n_val= 0.3, n_test= 0.1, neighborhood_steps= 2)

    model = Model(node_feature_dim = 500, gnn_in_features = 64, gnn_hidden_features = 64, gnn_out_features = 64)

    trainer = Model_Trainer(wikinetworkdata, label_info)
    model = trainer.train_model(dl, model, graph, n_epochs=20, early_stop=12, lr = 0.01, weight_decay=1e-5)
    trainer.evaluate_model(model, val_test_dl)


