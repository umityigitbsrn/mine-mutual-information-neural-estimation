# imports
import numpy as np
import torch
import torch.nn as nn
# noinspection PyPep8Naming
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

# constants
degree_of_poly_T = 5
num_of_poly_N = 100
prime_p = 2 ** 26 - 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# creating the dataset
## starting from basic shamir secret sharing with real numbers
## creating the polynomials first (reverse order - higher degree in the beginning)
first_poly_dataset = np.random.randint(1, prime_p, size=(num_of_poly_N, degree_of_poly_T),
                                       dtype=np.int64)
# to create a resulting polynomial without any zero in the largest degree
second_poly_dataset = np.random.randint(1, prime_p, size=(num_of_poly_N, degree_of_poly_T), dtype=np.int64)
multiplied_poly_dataset = np.empty((num_of_poly_N, 2 * (degree_of_poly_T - 1) + 1), dtype='object')
for idx in range(num_of_poly_N):
    multiplied_poly_dataset[idx] = np.polymul(first_poly_dataset[idx], second_poly_dataset[idx])

multiplied_poly_dataset = multiplied_poly_dataset % prime_p
mutual_information_dataset = np.empty((num_of_poly_N, 2 * (degree_of_poly_T - 1) + 3), dtype='object')
mutual_information_dataset[:, :(2 * (degree_of_poly_T - 1) + 1)] = multiplied_poly_dataset
mutual_information_dataset[:, -2] = second_poly_dataset[:, -1]
mutual_information_dataset[:, -1] = first_poly_dataset[:, -1]
mutual_information_dataset = np.fliplr(mutual_information_dataset)
mutual_information_dataset = mutual_information_dataset.astype(np.float64) / prime_p
mutual_information_dataset = torch.from_numpy(mutual_information_dataset.copy())


# sample from distributions for one dataset
def sample_for_iteration(batch_size, dataset):
    # sampling from joint distribution
    joint_idx = np.random.choice(dataset.shape[0], size=batch_size, replace=False)
    joint_batch = dataset[joint_idx]

    # sampling from marginal distributions
    ## first sample only for the secrets
    secret_marginal_idx = np.random.choice(dataset.shape[0], size=batch_size, replace=False)
    coeff_marginal_idx = np.random.choice(dataset.shape[0], size=batch_size, replace=False)
    marginal_batch = torch.empty((batch_size, dataset.shape[-1]), dtype=dataset.dtype)
    marginal_batch[:, :2] = dataset[secret_marginal_idx, :2]
    marginal_batch[:, 2:] = dataset[coeff_marginal_idx, 2:]

    return joint_batch, marginal_batch


class Mine(nn.Module):
    def __init__(self, input_size=2, hidden_size=100):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight, std=0.02)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, input_arg):
        output = F.relu(self.fc1(input_arg))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output


def mutual_information(joint, marginal, mine_net):
    t = mine_net(joint)
    et = torch.exp(mine_net(marginal))
    mi_lb = torch.mean(t) - torch.log(torch.mean(et))
    return mi_lb, t, et


def learn_mine(batch, mine_net, mine_net_optim, ma_et, ma_rate=0.01, device_arg=None):
    if device_arg is None:
        device_arg = 'cuda' if torch.cuda.is_available() else 'cpu'
    # batch is a tuple of (joint, marginal)
    joint, marginal = batch
    joint, marginal = joint.to(device_arg), marginal.to(device_arg)
    mi_lb, t, et = mutual_information(joint, marginal, mine_net)
    ma_et = (1 - ma_rate) * ma_et + ma_rate * torch.mean(et)

    # unbiasing use moving average
    loss = -(torch.mean(t) - (1 / ma_et.mean()).detach() * torch.mean(et))
    # use biased estimator
    # loss = - mi_lb

    mine_net_optim.zero_grad()
    autograd.backward(loss)
    mine_net_optim.step()
    return mi_lb, ma_et


def train(data, mine_net, mine_net_optim, batch_size=50, iter_num=int(5e+3), log_freq=100, device_arg=None):
    # data is x or y
    result = list()
    ma_et = 1.
    for i in range(iter_num):
        batch = sample_for_iteration(batch_size, data)
        mi_lb, ma_et = learn_mine(batch, mine_net, mine_net_optim, ma_et, device_arg=device_arg)
        result.append(mi_lb.detach().cpu().numpy())
        if (i + 1) % log_freq == 0:
            print(result[-1])
    return result


torch.set_default_dtype(torch.float64)
mine_net_sss = Mine(input_size=2 * (degree_of_poly_T - 1) + 3).to(device)
mine_net_optim_sss = optim.Adam(mine_net_sss.parameters(), lr=1e-3)
result_sss = train(mutual_information_dataset, mine_net_sss, mine_net_optim_sss)
