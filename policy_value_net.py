import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from constant import *

def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        # print(residual.size())
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # print(out.size())

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class policy_value_net(nn.Module):
    def __init__(self, block, inplanes, planes, stride=1):
        super(policy_value_net, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        blocks = []

        for i in range(NUM_BLOCK):
            blocks.append(block(planes, planes))

        self.layers = nn.Sequential(*blocks)

        # value head
        self.conv2 = nn.Conv2d(planes, 2, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(2)
        self.fc1 = nn.Linear(BOARD_SIZE ** 2 * 2, BOARD_SIZE ** 2)
        self.fc2 = nn.Linear(BOARD_SIZE ** 2, 1)


        # policy head
        self.conv3 = nn.Conv2d(planes, 2, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(2)
        self.fc3 = nn.Linear(BOARD_SIZE ** 2 * 2, (BOARD_SIZE - 1) ** 2 * 2+ 12)


    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layers(out)

        # value head
        value_out = self.conv2(out)
        value_out = self.bn2(value_out)
        value_out = self.relu(value_out)
        value_out = value_out.view(value_out.size(0), -1)
        value_out = self.fc1(value_out)
        value_out = F.tanh(self.fc2(value_out))

        # policy head
        policy_out = self.conv3(out)
        policy_out = self.bn3(policy_out)
        policy_out = self.relu(policy_out)
        policy_out = policy_out.view(policy_out.size(0), -1)
        # policy_out = F.log_softmax(self.fc3(policy_out), dim=1)  # softmax+log pytorch 0.4
        policy_out = F.log_softmax(self.fc3(policy_out), dim = 1)

        return policy_out,value_out

# device = torch.device("cpu")
# x = torch.Tensor(4,26,9,9).to(device)
# net = policy_value_net(BasicBlock,26,64).to(device)
# print(net)
# value_out,policy_out = net(x)
# print("value:",value_out.size())
# print("policy:",policy_out.size())


class PolicyValueNet(object):
    def __init__(self,model_file=None, use_gpu=True):
        self.use_gpu = use_gpu
        self.l2_const = 1e-4  # 
        if self.use_gpu:
            # device = torch.device("cuda:0")
            self.policy_value_net = policy_value_net(BasicBlock,6 + (WALL_NUM * 2),NN_DIM).cuda()
        else:
            # device = torch.device("cpu")
            self.policy_value_net = policy_value_net(BasicBlock,6 + (WALL_NUM * 2),NN_DIM)

        self.optimizer = optim.Adam(self.policy_value_net.parameters(), weight_decay=self.l2_const)

        if model_file:
            self.policy_value_net.load_state_dict(torch.load('ckpt/%s.pth'% model_file))

    def policy_value(self, state_batch):
        """
        """
        if self.use_gpu:
            # device = torch.device("cuda:0")
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs, value.data.cpu().numpy()
        else:
            device = torch.device("cpu")
            state_batch = Variable(torch.FloatTensor(state_batch))
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.numpy())
            return act_probs, value.data.numpy()

    def policy_value_fn(self, game):
        """
        """
        legal_positions = game.actions()  # 
        current_state = np.ascontiguousarray(game.state()).reshape([1, 6 + (WALL_NUM * 2), BOARD_SIZE, BOARD_SIZE])
        if self.use_gpu:
            # device = torch.device("cuda:0")
            log_act_probs, value = self.policy_value_net(Variable(torch.from_numpy(current_state)).cuda().float())
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        else:
            # device = torch.device("cpu")
            log_act_probs, value = self.policy_value_net(Variable(torch.from_numpy(current_state)).float())
            act_probs = np.exp(log_act_probs.data.numpy().flatten())
        # print("legal:",legal_positions)
        # print("probs:", act_probs)

        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.item()
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        if self.use_gpu:
            # device = torch.device("cuda:0")
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda())
            winner_batch = Variable(torch.FloatTensor(winner_batch).cuda())
        else:
            # device = torch.device("cpu")
            state_batch = Variable(torch.FloatTensor(state_batch))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs))
            winner_batch = Variable(torch.FloatTensor(winner_batch))

        self.optimizer.zero_grad()
        set_learning_rate(self.optimizer, lr)

        # 
        log_act_probs, value = self.policy_value_net(state_batch)
        
        value_loss = F.mse_loss(value.view(-1), winner_batch)

        # print(mcts_probs.shape, log_act_probs.shape)
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1))        

        loss = value_loss + policy_loss
        # 
        loss.backward()
        self.optimizer.step()
        #
        entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, 1))
        return value_loss.data, policy_loss.data, entropy.data  # Change code for newest PyTorch version

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        torch.save(self.policy_value_net.state_dict(), 'ckpt/%s.pth'%(model_file))


