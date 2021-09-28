from torch import nn
import torch


class EdgeLabelPredictor(nn.Module):

    def __init__(self, h_node_features, dot_product_dim = 4, include_edge_features = False):
        super().__init__()
        #self.W = nn.Linear(2 * in_features, num_classes)
        #self.W = nn.Linear(2 * in_features, 1) ## exchange this with siamese architecture

        self.node_fc1 = nn.Linear(h_node_features, dot_product_dim)
        self.include_edge_features = include_edge_features
        self.apply(self.init_weights)  ## weight initialisation


    def init_weights(self,m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.0)


    def encoder(self, node_emb):
        h1 = self.node_fc1(node_emb)
        return h1


    def bdot(self, a, b):
        ## batch dot product
        B = a.shape[0]
        S = a.shape[1]
        return torch.bmm(a.view(B, 1, S), b.view(B, S, 1)).squeeze(1)


    def mean_edge_node_feature(self, node_features, edge_features):
        #node_edge_mean = node_features - edge_features ## pairwise distance

        node_edge_mean = torch.stack((node_features, edge_features), 2)
        node_edge_mean = torch.mean(node_edge_mean, dim=2)
        return node_edge_mean


    def edge_wise_score(self, edges):

        #data = torch.cat([edges.src['x'], edges.dst['x']], dim =1) ## concat incident node features
        #print(dir(edges))
        #print(edges._edge_data)
        edge_features = edges.data["edge_h_features"]

        src_node = edges.src['node_h_features']
        dst_node = edges.dst['node_h_features']

        if self.include_edge_features:
            src_node = self.mean_edge_node_feature(src_node, edge_features)
            dst_node = self.mean_edge_node_feature(dst_node, edge_features)

        src_node = self.encoder(src_node)
        dst_node = self.encoder(dst_node)

        merge_out = self.bdot(src_node, dst_node)
        output_prob = torch.sigmoid(merge_out)

        edge_score_dict = {'label_score': output_prob}
        return edge_score_dict


    def forward(self, edge_subgraph, n, e):

        with edge_subgraph.local_scope():

            edge_subgraph.ndata['node_h_features'] = n ## assign hidden features from message passing to nodes
            edge_subgraph.edata['edge_h_features'] = e  ## assign hidden features from message passing to edges
            edge_subgraph.apply_edges(func = self.edge_wise_score, edges = '__ALL__')
            edge_labels_hat = edge_subgraph.edata['label_score']
            return edge_labels_hat