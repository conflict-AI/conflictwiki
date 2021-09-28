
from torch import nn

from wb0configs import configs
from wb4task.setting_transductive.archive.trans_dataload_class import load_data
from wb4task.setting_transductive.archive.trans_train_class import Trainer
from wb4task.models.pre_post_nns.edge_label_prediction_cat import EdgeLabelPredictor
from wb4task.models.pre_post_nns.feature_reduction import FeatureReducer
from wb4task.models.sage_model import SAGE
from wb4task.helper.train_helpers import get_edge_features, get_node_features



class Model(nn.Module):
    def __init__(self, node_feature_dim, gnn_in_features, gnn_hidden_features, gnn_out_features):
        super().__init__()

        self.node_feature_reducer = FeatureReducer(node_feature_dim, gnn_in_features)
        self.edge_feature_reducer = FeatureReducer(node_feature_dim, gnn_in_features)

        self.gnn = SAGE(gnn_in_features, gnn_hidden_features, gnn_out_features) ## message passing

        self.egde_label_predictor = EdgeLabelPredictor(gnn_out_features, include_edge_features= True) ## edge label computation


    def forward(self, g):
        # blocks is the graph
        # edge_subgraph is used for predictor only
        # x are node features

        n = self.node_feature_reducer(get_node_features(g, "node_features"))
        e = self.edge_feature_reducer(get_edge_features(g, "edge_features"))

        n = self.gnn(g, n, e)

        h = self.egde_label_predictor(g, n, e)
        return h



class Model_Trainer(Trainer):

    def __init__(self,wikinetworkdata, class_weights):
        super().__init__(wikinetworkdata, class_weights)


    def pass_data_to_model(self, model, train_step, graph):


        edge_predictions = model(graph)

        ## filter out non-masked edge labels
        label_mask = graph.edata['label_mask'].float()  ## get edge signs
        edge_mask = graph.edata[train_step + '_mask'] ## train, val, test filter

        if label_mask[label_mask == False].shape[0] > 0: ## random_label_frac < 1.0
            edge_mask[label_mask == True] = False   ## label mask

        edge_predictions = edge_predictions[edge_mask]
        edge_labels = graph.edata['label_discrete'].float()
        edge_labels = edge_labels[edge_mask].view(-1, 1)

        return edge_predictions, edge_labels



if __name__ == "__main__":

    config = configs.ConfigBase()
    graph, label_info, wikinetworkdata = load_data(config, n_val=0.15, n_test=0.15, random_node_frac = 0.0, random_label_frac = 0.0)

    model = Model(node_feature_dim = 500, gnn_in_features = 64, gnn_hidden_features = 64, gnn_out_features = 64)

    trainer = Model_Trainer(wikinetworkdata, label_info)
    model = trainer.train_model(model, graph, n_epochs=70, early_stop=10, lr = 0.01, weight_decay=1e-8)
    trainer.evaluate_model(model, graph)


