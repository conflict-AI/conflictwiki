
import torch
from torch import nn


# model definition
class Model(nn.Module):
    # define model elements
    def __init__(self, ):
        super(Model, self).__init__()

        ## encoder is shared across branches
        self.x_enc_h1 = nn.Linear(768, 256)
        self.x_enc_h2 = nn.Linear(256, 64)
        self.x_enc_h3 = nn.Linear(64, 8)


        ## encoder is shared across branches
        self.aux_enc_h1 = nn.Linear(768, 256)
        self.aux_enc_h2 = nn.Linear(256, 64)
        self.aux_enc_h3 = nn.Linear(64, 8)


    def init_weights(self,m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.0)

    def x_encoder(self, x):
        h1 = torch.relu(self.x_enc_h1(x))
        h2 = torch.relu(self.x_enc_h2(h1))
        h3 = self.x_enc_h3(h2)
        return h3

    def aux_encoder(self,e):
        h1 = torch.relu(self.aux_enc_h1(e))
        h2 = torch.relu(self.aux_enc_h2(h1))
        h3 = self.aux_enc_h3(h2)
        return h3

    def bdot(self, a, b):
        ## batch dot product
        B = a.shape[0]
        S = a.shape[1]
        return torch.bmm(a.view(B, 1, S), b.view(B, S, 1)).squeeze(1)

    def aux_merge(self, x, aux):
        return x - aux ## pairwise distance

    def x1_x2_merge(self, x_1, x_2):  ## dot product
        merge_out = self.bdot(x_1, x_2)
        return merge_out

    # forward propagate input
    def forward(self, x_1, x_2, aux):
        x_1 = self.x_encoder(x_1)
        x_2 = self.x_encoder(x_2)

        aux = self.aux_encoder(aux)
        x_1 = self.aux_merge(x_1, aux)
        x_2 = self.aux_merge(x_2, aux)

        merge_out = self.x1_x2_merge(x_1, x_2)
        output_prob = torch.sigmoid(merge_out)
        return output_prob



if __name__ == "__main__":

    model = Model()
    model_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model_param)