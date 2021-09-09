
import torch
from torch import nn
from torchsummary import summary

# model definition
class Model(nn.Module):
    # define model elements
    def __init__(self, ):
        super(Model, self).__init__()

        ## encoder is shared across branches
        self.x_e_enc_h1 = nn.Linear(768, 256)
        self.x_e_enc_h2 = nn.Linear(256, 64)
        self.x_e_enc_h3 = nn.Linear(64, 8)


        ## encoder is shared across branches
        self.x_c_enc_h1 = nn.Linear(768, 256)
        self.x_c_enc_h2 = nn.Linear(256, 64)
        self.x_c_enc_h3 = nn.Linear(64, 8)

        ## conflict encoder
        self.c_enc_h1 = nn.Linear(768, 256)
        self.c_enc_h2 = nn.Linear(256, 64)
        self.c_enc_h3 = nn.Linear(64, 8)

        ## entity encoder
        self.e_enc_h1 = nn.Linear(768, 256)
        self.e_enc_h2 = nn.Linear(256, 64)
        self.e_enc_h3 = nn.Linear(64, 8)


    def init_weights(self,m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.0)

    def x_e_encoder(self, x):
        h1 = torch.relu(self.x_e_enc_h1(x))
        h2 = torch.relu(self.x_e_enc_h2(h1))
        h3 = self.x_e_enc_h3(h2)
        return h3


    def x_c_encoder(self,x):
        h1 = torch.relu(self.x_c_enc_h1(x))
        h2 = torch.relu(self.x_c_enc_h2(h1))
        h3 = self.x_c_enc_h3(h2)
        return h3

    def e_encoder(self, x):
        h1 = torch.relu(self.e_enc_h1(x))
        h2 = torch.relu(self.e_enc_h2(h1))
        h3 = self.e_enc_h3(h2)
        return h3

    def c_encoder(self, x):
        h1 = torch.relu(self.c_enc_h1(x))
        h2 = torch.relu(self.c_enc_h2(h1))
        h3 = self.c_enc_h3(h2)
        return h3

    def bdot(self, a, b):
        ## batch dot product
        B = a.shape[0]
        S = a.shape[1]
        return torch.bmm(a.view(B, 1, S), b.view(B, S, 1)).squeeze(1)

    def x_aux_merge(self, x, y):
        return x - y ## pairwise distance

    def x1_x2_merge(self, x_1, x_2):  ## dot product
        merge_out = self.bdot(x_1, x_2)
        return merge_out

    # forward propagate input
    def forward(self, x_1_c, x_1_e, x_2_c, x_2_e, c, e):

        x_1_c = self.x_c_encoder(x_1_c)
        x_2_c = self.x_c_encoder(x_2_c)
        c = self.c_encoder(c)

        x_1_e = self.x_e_encoder(x_1_e)
        x_2_e = self.x_e_encoder(x_2_e)
        e = self.e_encoder(e)

        x_1_e = self.x_aux_merge(x_1_e, e)
        x_2_e = self.x_aux_merge(x_2_e, e)
        x_1_c = self.x_aux_merge(x_1_c, c)
        x_2_c = self.x_aux_merge(x_2_c, c)

        x_1 = self.x_aux_merge(x_1_e, x_1_c)
        x_2 = self.x_aux_merge(x_2_e, x_2_c)

        merge_out = self.x1_x2_merge(x_1, x_2)
        output_prob = torch.sigmoid(merge_out)
        return output_prob



if __name__ == "__main__":

    model = Model()
    model_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model_param)