import torch
import torch.nn as nn 
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(AutoEncoder, self).__init__()
        """
        784 -> 512 -> 256 -> 2 -> 256 -> 512 -> 784

        """
        #! encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc3 = nn.Linear(h_dim2, z_dim)
        #! decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)

    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)

    def decoder(self, x):
        h = F.relu(self.fc4(x))
        h = F.relu(self.fc5(h))
        return F.sigmoid(self.fc6(h))

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

if __name__ == "__main__" :
    ae = AutoEncoder(x_dim=784, h_dim1= 512, h_dim2=256, z_dim=2)
    inp = torch.rand(4 , 28*28)
    print(ae(inp).shape)