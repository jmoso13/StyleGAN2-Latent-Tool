import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def dim_decay_func(x, u, l, n):
  """
  Function for producing the dimensionalities of hidden layers
  
  Inputs
  ------
  x: Percentage step through the function, should be float value between 0 and 1
  u: Upper limit, where the dimensions start
  l: Lower limit, final dimensionality of z space
  n: How quickly the function decays, values above 1 are logarithmic while values below 1 are exponential
  """
  return int(np.rint(-np.power(x, n) * (u - l) + u))


class VAE(nn.Module):

    def __init__(self):
      super().__init__()
      self.fc1 = nn.Linear(512,506)
      self.fc2 = nn.Linear(506,487)
      self.fc3 = nn.Linear(487,455)
      self.fc4 = nn.Linear(455,411)
      self.fc5 = nn.Linear(411,355)
      self.fc6 = nn.Linear(355,285)
      self.fc7 = nn.Linear(285,203)
      self.fc8 = nn.Linear(203,109)

      self.fc_mu = nn.Linear(109, 2)
      self.fc_var = nn.Linear(109, 2)

      self.fcr8 = nn.Linear(2,109)
      self.fcr7 = nn.Linear(109,203)
      self.fcr6 = nn.Linear(203,285)
      self.fcr5 = nn.Linear(285,355)
      self.fcr4 = nn.Linear(355,411)
      self.fcr3 = nn.Linear(411,455)
      self.fcr2 = nn.Linear(455,487)
      self.fcr1 = nn.Linear(487,506)      
      self.fcr_out = nn.Linear(506,512)

      self.bn1 = nn.BatchNorm1d(506) 
      self.bn2 = nn.BatchNorm1d(487)
      self.bn3 = nn.BatchNorm1d(455)
      self.bn4 = nn.BatchNorm1d(411)
      self.bn5 = nn.BatchNorm1d(355)
      self.bn6 = nn.BatchNorm1d(285) 
      self.bn7 = nn.BatchNorm1d(203)
      self.bn8 = nn.BatchNorm1d(109)

      self.bnr1 = nn.BatchNorm1d(506) 
      self.bnr2 = nn.BatchNorm1d(487)
      self.bnr3 = nn.BatchNorm1d(455)
      self.bnr4 = nn.BatchNorm1d(411)
      self.bnr5 = nn.BatchNorm1d(355)
      self.bnr6 = nn.BatchNorm1d(285) 
      self.bnr7 = nn.BatchNorm1d(203)
      self.bnr8 = nn.BatchNorm1d(109)

    def encode(self, x):
      x = F.leaky_relu(self.bn1(self.fc1(x)))
      x = F.leaky_relu(self.bn2(self.fc2(x)))
      x = F.leaky_relu(self.bn3(self.fc3(x)))
      x = F.leaky_relu(self.bn4(self.fc4(x)))
      x = F.leaky_relu(self.bn5(self.fc5(x)))
      x = F.leaky_relu(self.bn6(self.fc6(x)))
      x = F.leaky_relu(self.bn7(self.fc7(x)))
      x = F.leaky_relu(self.bn8(self.fc8(x)))
      mu, log_var = self.fc_mu(x), self.fc_var(x)
      std = torch.exp(0.5 * log_var)
      q = torch.distributions.Normal(mu, std)
      z = q.rsample()

      return [z, mu, log_var]

    def decode(self, z):
      x = F.leaky_relu(self.bnr8(self.fcr8(z)))
      x = F.leaky_relu(self.bnr7(self.fcr7(x)))
      x = F.leaky_relu(self.bnr6(self.fcr6(x)))
      x = F.leaky_relu(self.bnr5(self.fcr5(x)))
      x = F.leaky_relu(self.bnr4(self.fcr4(x)))
      x = F.leaky_relu(self.bnr3(self.fcr3(x)))
      x = F.leaky_relu(self.bnr2(self.fcr2(x)))
      x = F.leaky_relu(self.bnr1(self.fcr1(x)))
      x = self.fcr_out(x)

      return x

    def forward(self, x):
      z, mu, log_var = self.encode(x)
      x = self.decode(z)

      return [x, mu, log_var]
  
