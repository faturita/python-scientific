"""
========================
Stable Diffusion Example
========================

Train a basic diffusion model to approximate an empirical distribution

# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/

# Written by Francois Fleuret <francois@fleuret.org>

"""
print(__doc__)

import math, argparse

import matplotlib.pyplot as plt

import torch, torchvision
from torch import nn
from torch.nn import functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'device {device}')

######################################################################

def sample_gaussian_mixture(nb):
    p, std = 0.3, 0.2
    result = torch.randn(nb, 1) * std
    result = result + torch.sign(torch.rand(result.size()) - p) / 2
    return result

def sample_ramp(nb):
    result = torch.min(torch.rand(nb, 1), torch.rand(nb, 1))
    return result

def sample_two_discs(nb):
    a = torch.rand(nb) * math.pi * 2
    b = torch.rand(nb).sqrt()
    q = (torch.rand(nb) <= 0.5).long()
    b = b * (0.3 + 0.2 * q)
    result = torch.empty(nb, 2)
    result[:, 0] = a.cos() * b - 0.5 + q
    result[:, 1] = a.sin() * b - 0.5 + q
    return result

def sample_disc_grid(nb):
    a = torch.rand(nb) * math.pi * 2
    b = torch.rand(nb).sqrt()
    N = 4
    q = (torch.randint(N, (nb,)) - (N - 1) / 2) / ((N - 1) / 2)
    r = (torch.randint(N, (nb,)) - (N - 1) / 2) / ((N - 1) / 2)
    b = b * 0.1
    result = torch.empty(nb, 2)
    result[:, 0] = a.cos() * b + q
    result[:, 1] = a.sin() * b + r
    return result

def sample_spiral(nb):
    u = torch.rand(nb)
    rho = u * 0.65 + 0.25 + torch.rand(nb) * 0.15
    theta = u * math.pi * 3
    result = torch.empty(nb, 2)
    result[:, 0] = theta.cos() * rho
    result[:, 1] = theta.sin() * rho
    return result

def sample_mnist(nb):
    train_set = torchvision.datasets.MNIST(root = './data/', train = True, download = True)
    result = train_set.data[:nb].to(device).view(-1, 1, 28, 28).float()
    return result

samplers = {
    f.__name__.removeprefix('sample_') : f for f in [
        sample_gaussian_mixture,
        sample_ramp,
        sample_two_discs,
        sample_disc_grid,
        sample_spiral,
        sample_mnist,
    ]
}

######################################################################

parser = argparse.ArgumentParser(
    description = '''A minimal implementation of Jonathan Ho, Ajay Jain, Pieter Abbeel
"Denoising Diffusion Probabilistic Models" (2020)
https://arxiv.org/abs/2006.11239''',

    formatter_class = argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--seed',
                    type = int, default = 0,
                    help = 'Random seed, < 0 is no seeding')

parser.add_argument('--nb_epochs',
                    type = int, default = 100,
                    help = 'How many epochs')

parser.add_argument('--batch_size',
                    type = int, default = 25,
                    help = 'Batch size')

parser.add_argument('--nb_samples',
                    type = int, default = 25000,
                    help = 'Number of training examples')

parser.add_argument('--learning_rate',
                    type = float, default = 1e-3,
                    help = 'Learning rate')

parser.add_argument('--ema_decay',
                    type = float, default = 0.9999,
                    help = 'EMA decay, <= 0 is no EMA')

data_list = ', '.join( [ str(k) for k in samplers ])

parser.add_argument('--data',
                    type = str, default = 'gaussian_mixture',
                    help = f'Toy data-set to use: {data_list}')

parser.add_argument('--no_window',
                    action='store_true', default = False)

args = parser.parse_args()

if args.seed >= 0:
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

######################################################################

class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.mem = { }
        with torch.no_grad():
            for p in model.parameters():
                self.mem[p] = p.clone()

    def step(self):
        with torch.no_grad():
            for p in self.model.parameters():
                self.mem[p].copy_(self.decay * self.mem[p] + (1 - self.decay) * p)

    def copy_to_model(self):
        with torch.no_grad():
            for p in self.model.parameters():
                p.copy_(self.mem[p])

######################################################################

# Gets a pair (x, t) and appends t (scalar or 1d tensor) to x as an
# additional dimension / channel

class TimeAppender(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, u):
        x, t = u
        if not torch.is_tensor(t):
            t = x.new_full((x.size(0),), t)
        t = t.view((-1,) + (1,) * (x.dim() - 1)).expand_as(x[:,:1])
        return torch.cat((x, t), 1)

class ConvNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        ks, nc = 5, 64

        self.core = nn.Sequential(
            TimeAppender(),
            nn.Conv2d(in_channels + 1, nc, ks, padding = ks//2),
            nn.ReLU(),
            nn.Conv2d(nc, nc, ks, padding = ks//2),
            nn.ReLU(),
            nn.Conv2d(nc, nc, ks, padding = ks//2),
            nn.ReLU(),
            nn.Conv2d(nc, nc, ks, padding = ks//2),
            nn.ReLU(),
            nn.Conv2d(nc, nc, ks, padding = ks//2),
            nn.ReLU(),
            nn.Conv2d(nc, out_channels, ks, padding = ks//2),
        )

    def forward(self, u):
        return self.core(u)

######################################################################
# Data

try:
    train_input = samplers[args.data](args.nb_samples).to(device)
except KeyError:
    print(f'unknown data {args.data}')
    exit(1)

train_mean, train_std = train_input.mean(), train_input.std()

######################################################################
# Model

if train_input.dim() == 2:
    nh = 256

    model = nn.Sequential(
        TimeAppender(),
        nn.Linear(train_input.size(1) + 1, nh),
        nn.ReLU(),
        nn.Linear(nh, nh),
        nn.ReLU(),
        nn.Linear(nh, nh),
        nn.ReLU(),
        nn.Linear(nh, train_input.size(1)),
    )

elif train_input.dim() == 4:

    model = ConvNet(train_input.size(1), train_input.size(1))

model.to(device)

print(f'nb_parameters {sum([ p.numel() for p in model.parameters() ])}')

######################################################################
# Generate

def generate(size, T, alpha, alpha_bar, sigma, model, train_mean, train_std):

    with torch.no_grad():

        x = torch.randn(size, device = device)

        for t in range(T-1, -1, -1):
            output = model((x, t / (T - 1) - 0.5))
            z = torch.zeros_like(x) if t == 0 else torch.randn_like(x)
            x = 1/torch.sqrt(alpha[t]) \
                * (x - (1-alpha[t]) / torch.sqrt(1-alpha_bar[t]) * output) \
                + sigma[t] * z

        x = x * train_std + train_mean

        return x

######################################################################
# Train

T = 1000
beta = torch.linspace(1e-4, 0.02, T, device = device)
alpha = 1 - beta
alpha_bar = alpha.log().cumsum(0).exp()
sigma = beta.sqrt()

ema = EMA(model, decay = args.ema_decay) if args.ema_decay > 0 else None

for k in range(args.nb_epochs):

    acc_loss = 0
    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)

    for x0 in train_input.split(args.batch_size):
        x0 = (x0 - train_mean) / train_std
        t = torch.randint(T, (x0.size(0),) + (1,) * (x0.dim() - 1), device = x0.device)
        eps = torch.randn_like(x0)
        xt = torch.sqrt(alpha_bar[t]) * x0 + torch.sqrt(1 - alpha_bar[t]) * eps
        output = model((xt, t / (T - 1) - 0.5))
        loss = (eps - output).pow(2).mean()
        acc_loss += loss.item() * x0.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ema is not None: ema.step()

    print(f'{k} {acc_loss / train_input.size(0)}')

if ema is not None: ema.copy_to_model()

######################################################################
# Plot

model.eval()

########################################
# Nx1 -> histogram
if train_input.dim() == 2 and train_input.size(1) == 1:

    fig = plt.figure()
    fig.set_figheight(5)
    fig.set_figwidth(8)

    ax = fig.add_subplot(1, 1, 1)

    x = generate((10000, 1), T, alpha, alpha_bar, sigma,
                 model, train_mean, train_std)

    ax.set_xlim(-1.25, 1.25)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)

    d = train_input.flatten().detach().to('cpu').numpy()
    ax.hist(d, 25, (-1, 1),
            density = True,
            histtype = 'bar', edgecolor = 'white', color = 'lightblue', label = 'Train')

    d = x.flatten().detach().to('cpu').numpy()
    ax.hist(d, 25, (-1, 1),
            density = True,
            histtype = 'step', color = 'red', label = 'Synthesis')

    ax.legend(frameon = False, loc = 2)

    filename = f'minidiffusion_{args.data}.pdf'
    print(f'saving {filename}')
    fig.savefig(filename, bbox_inches='tight')

    if not args.no_window and hasattr(plt.get_current_fig_manager(), 'window'):
        plt.get_current_fig_manager().window.setGeometry(2, 2, 1024, 768)
        plt.show()

########################################
# Nx2 -> scatter plot
elif train_input.dim() == 2 and train_input.size(1) == 2:

    fig = plt.figure()
    fig.set_figheight(6)
    fig.set_figwidth(6)

    ax = fig.add_subplot(1, 1, 1)

    x = generate((1000, 2), T, alpha, alpha_bar, sigma,
                 model, train_mean, train_std)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set(aspect = 1)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)

    d = train_input[:x.size(0)].detach().to('cpu').numpy()
    ax.scatter(d[:, 0], d[:, 1],
               s = 2.5, color = 'gray', label = 'Train')

    d = x.detach().to('cpu').numpy()
    ax.scatter(d[:, 0], d[:, 1],
               s = 2.0, color = 'red', label = 'Synthesis')

    ax.legend(frameon = False, loc = 2)

    filename = f'minidiffusion_{args.data}.pdf'
    print(f'saving {filename}')
    fig.savefig(filename, bbox_inches='tight')

    if not args.no_window and hasattr(plt.get_current_fig_manager(), 'window'):
        plt.get_current_fig_manager().window.setGeometry(2, 2, 1024, 768)
        plt.show()

########################################
# NxCxHxW -> image
elif train_input.dim() == 4:

    x = generate((128,) + train_input.size()[1:], T, alpha, alpha_bar, sigma,
                 model, train_mean, train_std)

    x = torchvision.utils.make_grid(x.clamp(min = 0, max = 255),
                                    nrow = 16, padding = 1, pad_value = 64)
    x = F.pad(x, pad = (2, 2, 2, 2), value = 64)[None]

    t = torchvision.utils.make_grid(train_input[:128],
                                    nrow = 16, padding = 1, pad_value = 64)
    t = F.pad(t, pad = (2, 2, 2, 2), value = 64)[None]

    result = 1 - torch.cat((t, x), 2) / 255

    filename = f'minidiffusion_{args.data}.png'
    print(f'saving {filename}')
    torchvision.utils.save_image(result, filename)

else:

    print(f'cannot plot result of size {train_input.size()}')

######################################################################