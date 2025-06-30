import torch
import torch.nn as nn
import torch.optim as optim 
from torchvision import datasets , transforms

from model import AutoEncoder
# !----------------------------------------------
# ? Hyper Parameters 
# !----------------------------------------------
device = "cuda"if torch.cuda.is_available() else "cpu"
batch_size = 128 
alpha = 0.0003
epochs = 30 

input_dim = 28*28
h1 = 512
h2 = 256
z_dim = 2

# !----------------------------------------------

data_transform = transforms.Compose([
    transforms.ToTensor() , 
    lambda x :x.view(-1) , 
])

train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform = data_transform, download=True)
test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform= data_transform, download=False)


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# ! Defining The Model 

model = AutoEncoder(x_dim=input_dim, h_dim1= h1, h_dim2=h2, z_dim=z_dim).to(device)

optimizer = optim.Adam(model.parameters() , lr=alpha)
loss_f =nn.BCELoss(reduction="sum")

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        
        optimizer.zero_grad()
        data= data.to(device)

        recon_batch = model(data)
        loss = loss_f(recon_batch, data)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))



def test():
    model.eval()
    test_loss= 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            
            recon= model(data)

            # sum up batch loss
            test_loss += loss_f(recon, data).item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, epochs+1):
    train(epoch)
    test()


torch.save(model.state_dict(), 'AE.pth')