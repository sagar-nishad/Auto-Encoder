import torch 
from model import AutoEncoder
from torch.autograd import Variable
from torchvision.utils import save_image

model = AutoEncoder(x_dim=784, h_dim1= 512, h_dim2=256, z_dim=2)
sd = torch.load("./AE.pth")
model.load_state_dict(sd)

with torch.no_grad():
    z = torch.randn(64, 2)
    sample = model.decoder(z)
    save_image(sample.view(64, 1, 28, 28), './sample' + '.png')