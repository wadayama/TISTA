# Trainable ISTA (TISTA)
# 
# This code is an implementation of Trainable ISTA (TISTA) for sparse signal recovery in Pytorch.Tensor.
# The details of the algorithm can be found in the paper:
# Daisuke Ito, Satoshi Takabe, Tadashi Wadayama,
# "Trainable ISTA for Sparse Signal Recovery", arXiv:1801.01978.
# (Computer experiments in the paper was performed with another TensorFlow implementation)
#
# GPU is required for execution of this program. If you do not have GPU,
# just change "device = torch.device('cuda')" to 'cpu'.
# 
# This basic TISTA trains only $\gamma_t$.
#
# Last update 11/21/2018

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import math
import time

# device
device = torch.device('cuda') # choose 'cpu' or 'cuda'

# global variables

N = 500  # length of a source signal vector
M = 250  # length of a observation vector
p = 0.1  # probability for occurrence of non-zero components

batch_size = 1000  # mini-batch size
num_batch = 200  # number of mini-batches in a generation
num_generations = 12  # number of generations
snr = 40.0  # SNR for the system in dB

alpha2 = 1.0  # variance of non-zero component
alpha_std = math.sqrt(alpha2)
max_layers = 12  # maximum number of layers
adam_lr = 0.04  # initial learning parameter for Adam

# random seed of torch
torch.manual_seed(5)

### setting sensing matrix
# sensing matrix with small variance
A = torch.normal(0.0, std=math.sqrt(1.0/M) * torch.ones(M, N)) 

# sensing matrix with large variance
#A = torch.normal(0.0, std=math.sqrt(1.0) * torch.ones(M, N))  

# \pm 1 sensing matrix 
#A = 1.0-2.0*torch.bernoulli(0.5 * torch.ones(M, N))  
### end of setting sensing matrix

At = A.t()
W = At.mm((A.mm(At)).inverse())  # pseudo inverse matrix
Wt = W.t()

taa = (At.mm(A)).trace().to(device)  # trace(A^T A)
tww = (W.mm(Wt)).trace().to(device)  # trace(W W^T)

Wt = torch.Tensor(Wt).to(device)
At = torch.Tensor(At).to(device)

print("sensing matrix A\n", A.detach().numpy())


# detection for NaN
def isnan(x):
    return x != x

# mini-batch generator
def generate_batch():
    support = torch.bernoulli(p * torch.ones(batch_size, N))
    nonzero = torch.normal(0.0, alpha_std * torch.ones(batch_size, N))
    return torch.mul(nonzero, support)


# definition of TISTA network
class TISTA_NET(nn.Module): 
    def __init__(self):
        super(TISTA_NET, self).__init__() 
        self.gamma = nn.Parameter(torch.ones(max_layers)) #nn.Parameter(torch.normal(1.0, 0.1*torch.ones(max_layers))) 
        print("TISTA initialized...")

    def gauss(self, x,  var):
        return torch.exp(-torch.mul(x, x)/(2.0*var))/pow(2.0*math.pi*var,0.5)

    def MMSE_shrinkage(self, y, tau2):  # MMSE shrinkage function
        return (y*alpha2/(alpha2+tau2))*p*self.gauss(y,(alpha2+tau2))/((1-p)*self.gauss(y, tau2) + p*self.gauss(y, (alpha2+tau2)))

    def eval_tau2(self, t, i):  # error variance estimator
        v2 = (t.norm(2,1).pow(2.0) - M*sigma2)/taa
        v2.clamp(min=1e-9)
        tau2 = (v2/N)*(N+(self.gamma[i]*self.gamma[i]-2.0*self.gamma[i])*M)+self.gamma[i]*self.gamma[i]*tww*sigma2/N
        tau2 = (tau2.expand(N, batch_size)).t()
        return tau2
        
    def forward(self, x, s, max_itr):  # TISTA network
        y = x.mm(At) + torch.Tensor(torch.normal(0.0, sigma_std*torch.ones(batch_size, M))).to(device)
        for i in range(max_itr):
            t = y - s.mm(At)
            tau2 = self.eval_tau2(t, i)
            r = s + t.mm(Wt)*self.gamma[i]
            s = self.MMSE_shrinkage(r, tau2)
        return s



global sigma_std, sigma2, xi

network = TISTA_NET().to(device)  # generating an instance of TISTA network
s_zero = torch.Tensor(torch.zeros(batch_size, N)).to(device)  # initial value
opt = optim.Adam(network.parameters(), lr=adam_lr)  # setting for optimizer (Adam)

# SNR calculation
sum = 0.0
for i in range(100):
    x = torch.Tensor(generate_batch()).to(device)
    y = x.mm(At)
    sum += (y.norm(2, 1).pow(2.0)).sum().item()
ave = sum/(100.0 * batch_size)
sigma2 = ave/(M*math.pow(10.0, snr/10.0))
sigma_std = math.sqrt(sigma2)
xi = alpha2 + sigma2


# incremental training loop
start = time.time()

for gen in (range(num_generations)):
    # training process  
    for i in range(num_batch):
        if (gen > 10): # change learning rate of Adam
           opt = optim.Adam(network.parameters(), lr=adam_lr/50.0)
        x = torch.Tensor(generate_batch()).to(device)
        opt.zero_grad()
        x_hat = network(x, s_zero, gen+1).to(device)
        loss = F.mse_loss(x_hat, x)
        loss.backward()

        grads = torch.stack([param.grad for param in network.parameters()])
        if isnan(grads).any():  # avoiding NaN in gradients
            continue


        opt.step()
    # end of training training


    # accuracy check after t-th incremental training
    nmse_sum = 0.0
    tot = 1 # batch size for accuracy check
    for i in range(tot):
        x = torch.Tensor(generate_batch()).to(device)
        x_hat = network(x, s_zero, gen+1).to(device)
        num = (x - x_hat).norm(2, 1).pow(2.0)
        denom = x.norm(2,1).pow(2.0)
        nmse = num/denom
        nmse_sum += torch.sum(nmse).item()
    
    nmse = 10.0*math.log(nmse_sum / (tot * batch_size))/math.log(10.0) #NMSE [dB]

    print('({0}) NMSE= {1:6.3f}'.format(gen + 1, nmse))
    # end of accuracy check

elapsed_time = time.time() - start
print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
