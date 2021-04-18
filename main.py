import torch
from torch.nn import Module, Conv1d, ConvTranspose1d, BatchNorm1d, CrossEntropyLoss, Embedding
from torch import optim as optimiser

torch.manual_seed(0)


def num(x, d):
    for i in range(0, len(d)):
        if x==d[i]:
            return i

#Vanilla autoencoder used as basis of VAE
class Encoder(Module):
    def __init__(self, n):
        super(Encoder, self).__init__()
        self.embd = Embedding(n, 1024)
        self.cnn1 = Conv1d(1024, 1024, kernel_size = 3, stride = 3)
        self.cnn2 = Conv1d(1024, 1024, kernel_size = 5, stride = 5)
        self.cnn1_g = Conv1d(1024, 1024, kernel_size = 3, stride = 3)
        self.cnn2_g = Conv1d(1024, 1024, kernel_size = 5, stride = 5)
        self.bn1 = BatchNorm1d(1024)
        self.bn2 = BatchNorm1d(1024)
    def forward(self, x):
        y = self.embd(x).permute(0,2,1)
        y = self.bn1(y)
        y = self.cnn1(y)*self.cnn1_g(y).sigmoid()
        y = self.bn2(y)
        y = self.cnn2(y)*self.cnn2_g(y).sigmoid()
        return y


class Decoder(Module):
    def __init__(self, n):
        super(Decoder, self).__init__()
        self.cnn2 = ConvTranspose1d(1024, 1024, kernel_size = 5, stride = 5)
        self.cnn3 = ConvTranspose1d(1024, 1024, kernel_size = 3, stride = 3)
        self.cnn4 = Conv1d(1024, 1024, kernel_size = 3, padding = 1)
        self.cnn5 = Conv1d(1024, 1024, kernel_size = 3, padding = 1)
        self.cnn6 = Conv1d(1024, 1024, kernel_size = 3, padding = 1)
        self.cnn7 = Conv1d(1024, n, kernel_size = 1)
        self.cnn2_g = ConvTranspose1d(1024, 1024, kernel_size = 5, stride = 5)
        self.cnn3_g = ConvTranspose1d(1024, 1024, kernel_size = 3, stride = 3)
        self.cnn4_g = Conv1d(1024, 1024, kernel_size = 3, padding = 1)
        self.cnn5_g = Conv1d(1024, 1024, kernel_size = 3, padding = 1)
        self.cnn6_g = Conv1d(1024, 1024, kernel_size = 3, padding = 1)
        self.cnn7_g = Conv1d(1024, n, kernel_size = 1)
        self.bn2 = BatchNorm1d(1024)
        self.bn3 = BatchNorm1d(1024)
        self.bn4 = BatchNorm1d(1024)
        self.bn5 = BatchNorm1d(1024)
        self.bn6 = BatchNorm1d(1024)
        self.bn7 = BatchNorm1d(1024)
    def forward(self, x):
        y = x
        y = self.bn2(y)
        y = self.cnn2(y)*self.cnn2_g(y).sigmoid()
        y = self.bn3(y)
        y = self.cnn3(y)*self.cnn3_g(y).sigmoid()
        y = self.bn4(y)
        y = self.cnn4(y)*self.cnn4_g(y).sigmoid()
        y = self.bn5(y)
        y = self.cnn5(y)*self.cnn5_g(y).sigmoid()
        y = self.bn6(y)
        y = self.cnn6(y)*self.cnn6_g(y).sigmoid()
        y = self.bn7(y)
        y = self.cnn7(y)*self.cnn7_g(y).sigmoid()
        return y

class Autoencoder(Module):
    def __init__(self, n):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(n)
        self.decoder = Decoder(n)
    def forward(self, x):
        y = x
        y = self.encoder(y)
        y = self.decoder(y)
        return y


x0 = torch.load("code.dat")#Codes of words from biography
d = torch.load("dict.dat")#Words from biography, in order of their codes.


loss_fn = CrossEntropyLoss()
VAE = Autoencoder(len(d))

optim = optimiser.RMSprop(VAE.parameters(), lr = 1e-4, centered = True)


corr = 0.0
for i in range(0, 1000):#Training procedure
    x = []
    n = (torch.rand(100)*(x0.size(-1) - 15)).long()
    for j in range(0, 100):
        x.append(x0[:, n[j]:(n[j]+15)])
    x = torch.cat(x, dim = 0)
    y = VAE(x)
    loss = loss_fn(y, x)
    corr = y.max(dim = 1)[1].eq(x).float().mean()
    print(corr.item())
    loss.backward()
    optim.step()
    VAE.zero_grad()
torch.save(VAE, "Autoencoder.pt")
