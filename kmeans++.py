import torch
from torch.nn import Module, Conv1d, ConvTranspose1d, BatchNorm1d, Embedding



def num(x, d):
    for i in range(0, len(d)):
        if x==d[i]:
            return i


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



class Cluster_kmeans_pp(Module):
    def __init__(self, vae, x, n):
        super(Cluster_kmeans_pp, self).__init__()
        with torch.no_grad():
            z = vae.encoder(x)
            z_ = z[0].unsqueeze(0)
            for i in range(0, n-1):
                p = z.unsqueeze(0) - z_.unsqueeze(1)
                p = (p*p).sum(dim = 2).sum(dim = 2).min(dim = 0)[0].sqrt()
                cat = torch.distributions.Categorical(p)
                t = cat.sample([1])
                z1 = z[t]
                z_ = torch.cat([z_, z1], dim = 0)
            self.m = z_
            self.sd = torch.ones(z_.size())
            self.p = torch.zeros(n) + 1e-6
            self.i = 0
            self.vae = vae
    def cluster(self, x):
        with torch.no_grad():
            y = self.vae.encoder(x)
            z = y.unsqueeze(1) - self.m.unsqueeze(0)
            z = (z*z).sum(dim = 2).sum(dim = 2).min(dim = 1)[1]
            for i in range(0, z.size(0)):
                self.p[z[i]] = self.p[z[i]] + 1.0
                self.m[z[i]] = self.m[z[i]]*0.001 + y[i]*0.999
                self.sd[z[i]] = ((self.m[z[i]] - y[i])**2)*0.001 + self.sd[z[i]]*0.999
            self.i = self.i + 1
            return self.i
    def gen(self, n):
        with torch.no_grad():
            cat = torch.distributions.Categorical(self.p)
            z = cat.sample([n])
            mu = self.m[z]
            sd = self.sd[z]
            norm = torch.distributions.Normal(mu, sd.sqrt())
            y = norm.sample()
            y = torch.split(y, 1, dim = 0)
            y = torch.cat(y, dim = 2)
            y = self.vae.decoder(y).max(dim = 1)[1]
            return y



x0 = torch.load("code.dat")
d = torch.load("dict.dat")
with torch.no_grad():
    x = []
    n = (torch.rand(100)*(x0.size(1) - 15)).long()
    for i in range(0, 100):
        x.append(x0[:, n[i]:(n[i] + 15)])
    x = torch.cat(x, dim = 0)


VAE = torch.load("Autoencoder.pt")
VAE.eval()

clstr = Cluster_kmeans_pp(VAE, x, 20)
clstr.eval()

for i in range(0, 10000):
    x = []
    n = (torch.rand(10)*(x0.size(-1) - 15)).long()
    for j in range(0, 10):
        x.append(x0[:, n[j]:(n[j]+15)])
    x = torch.cat(x, dim = 0)
    y = clstr.cluster(x)
    print(y)

torch.save(clstr, "Clusters.pt")