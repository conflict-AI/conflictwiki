
from torch import nn
import torch
torch.manual_seed(0)

from torch.nn import functional as F
from dgl import nn as dglnn
import dgl
#from torch_geometric.nn.conv.signed_conv import SignedConv

from wb0configs import configs
from wb4task.setting_transductive.trans_batched_dataload_class import load_data
from wb4task.setting_transductive.trans_batched_train_class import Trainer

from wb4task.models.pre_post_nns.edge_label_prediction_mean import EdgeLabelPredictor
from wb4task.models.pre_post_nns.feature_reduction import FeatureReducer
from wb4task.models.gin_model import SignedGIN
from wb4task.helper.load_helpers import randomise_labels, randomise_node_features, randomise_edge_features


class Model(nn.Module):

    def __init__(self, node_feature_dim, neighborhood_steps, gnn_in_features, gnn_hidden_features, gnn_out_features, dot_product_dim, include_edge_features):
        super().__init__()

        self.node_feature_reducer = FeatureReducer(node_feature_dim, gnn_in_features)
        self.edge_feature_reducer = FeatureReducer(node_feature_dim, gnn_in_features)

        self.gnn = SignedGIN(neighborhood_steps, gnn_in_features, gnn_hidden_features, gnn_out_features) ## message passing

        self.egde_label_predictor = EdgeLabelPredictor(gnn_out_features, dot_product_dim, include_edge_features= include_edge_features) ## edge label computation



    def forward(self, edge_subgraph, blocks):
        # blocks is the graph
        # edge_subgraph is used for predictor only
        # x are node features

        n = self.gnn(self.node_feature_reducer, self.edge_feature_reducer, blocks)

        e = self.edge_feature_reducer(edge_subgraph.edata['edge_features'])
        h = self.egde_label_predictor(edge_subgraph, n, e)
        return h




class Model_Trainer(Trainer):

    def __init__(self,wikinetworkdata, class_weights, task_setting):
        super().__init__(wikinetworkdata, class_weights, task_setting)


    def pass_data_to_model(self, model, train_step, edge_subgraph, blocks):

        blocks = self.setup_task(train_step, edge_subgraph, blocks)
        edge_predictions = model(edge_subgraph, blocks)
        edge_labels = edge_subgraph.edata['label_discrete'].float()

        edge_mask = edge_subgraph.edata[train_step + '_mask']
        edge_predictions = edge_predictions[edge_mask]
        edge_labels = edge_labels[edge_mask].view(-1, 1)

        return edge_predictions, edge_labels



    def setup_task(self, train_step, subgraph, blocks):

        k_steps = len(blocks)

        ## randomise data in block
        dyad_node_ids = blocks[k_steps - 1].dstdata["_ID"]

        edge_mask = subgraph.edata[train_step + '_mask']
        dyad_edge_ids = subgraph.edata["_ID"][edge_mask]

        random_node_features = randomise_node_features(return_n=len(dyad_node_ids))
        random_edge_features = randomise_edge_features(return_n=len(dyad_edge_ids))

        for k in range(0, k_steps):
            dyad_node_indeces_src = (blocks[k].srcdata['_ID'].unsqueeze(1) == dyad_node_ids).nonzero(as_tuple=False)[:,0]
            dyad_node_indeces_dst = (blocks[k].dstdata['_ID'].unsqueeze(1) == dyad_node_ids).nonzero(as_tuple=False)[:,0]
            dyad_edge_indeces = (blocks[k].edata['_ID'].unsqueeze(1) == dyad_edge_ids).nonzero(as_tuple=False)[:,0]

            if self.task_setting == "systemic":
                blocks[k].srcdata["node_features"][dyad_node_indeces_src] = random_node_features.squeeze() ## randomise block
                blocks[k].dstdata["node_features"][dyad_node_indeces_dst] = random_node_features.squeeze() ## randomise block
                blocks[k].edata["edge_features"][dyad_edge_indeces] = random_edge_features.squeeze()  ## randomise block
            blocks[k].edata['label_mask'][dyad_edge_indeces] = True  ## randomise block

        #if self.task_setting == "systemic":
        #    randomise_node_features(subgraph, random_node_frac=1.0)  ## randomise data in edge subgraph
        #    randomise_edge_features(subgraph, random_edge_frac=1.0)  ## randomise data in edge subgraph
        #randomise_labels(subgraph, random_label_frac = 1.0)  ## randomise block
        return blocks


if __name__ == "__main__":

    config = configs.ConfigBase()
    train_dl, val_dl, test_dl, graph, label_info, wikinetworkdata = load_data(config, batch_size = 512, n_val= 0.3, n_test= 0.1, neighborhood_steps= 2, random_node_frac = 0.0, random_edge_frac = 0.0, random_label_frac = 0.0)

    model = Model(node_feature_dim = 500, neighborhood_steps= 2, gnn_in_features = 64, gnn_hidden_features = 64, gnn_out_features = 64, dot_product_dim = 12, include_edge_features = True)

    trainer = Model_Trainer(wikinetworkdata, label_info, task_setting = "systemic")
    model = trainer.train_model(train_dl, val_dl, model, graph, n_epochs=30, early_stop=3, lr = 0.001, weight_decay=1e-5)
    trainer.evaluate_model(model, test_dl)


