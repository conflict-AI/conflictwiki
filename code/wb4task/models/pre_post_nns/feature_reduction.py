from torch import nn
import torch


class FeatureReducer(nn.Module):

    def __init__(self, node_feature_dim, gnn_in_features):
        super().__init__()
        #self.W = nn.Linear(2 * in_features, num_classes)
        #self.W = nn.Linear(2 * in_features, 1) ## exchange this with siamese architecture

        self.node_fc1 = nn.Linear(node_feature_dim, 256)
        self.node_fc2 = nn.Linear(256, 64)
        self.node_fc3 = nn.Linear(64, gnn_in_features)

        self.apply(self.init_weights)  ## weight initialisation


    def init_weights(self,m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.0)


    def encoder(self, x):
        x = x.squeeze(1).float()
        h1 = torch.relu(self.node_fc1(x))
        h2 = torch.relu(self.node_fc2(h1))
        h3 = torch.relu(self.node_fc3(h2))
        return h3


    def forward(self, node_emb):
        node_low_dim_emb = self.encoder(node_emb)
        return node_low_dim_emb