import torch
from torch import nn


# model definition
class Model(nn.Module):
    # define model elements
    def __init__(self, ):
        super(Model, self).__init__()

        ## encoder is shared across branches
        self.enc_h1 = nn.Linear(768, 256)
        self.enc_h2 = nn.Linear(256, 64)
        self.enc_h3 = nn.Linear(64, 12)


    def init_weights(self,m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.0)

    def encoder(self, x):
        h1 = torch.relu(self.enc_h1(x))
        h2 = torch.relu(self.enc_h2(h1))
        h3 = self.enc_h3(h2)
        return h3

    def bdot(self, a, b):
        ## batch dot product
        B = a.shape[0]
        S = a.shape[1]
        return torch.bmm(a.view(B, 1, S), b.view(B, S, 1)).squeeze(1)

    def x1_x2_merge(self, x_1, x_2):  ## dot product
        merge_out = self.bdot(x_1, x_2)
        return merge_out

    # forward propagate input
    def forward(self, x_1, x_2):
        x_1 = self.encoder(x_1)
        x_2 = self.encoder(x_2)
        merge_out = self.x1_x2_merge(x_1, x_2)
        output_prob = torch.sigmoid(merge_out)
        return output_prob


if __name__ == "__main__":

    model = Model()
    model_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model_param)