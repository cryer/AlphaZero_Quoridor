import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

def set_learning_rate(optimizer, lr):
    """设置学习率"""
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
        self.res1 = block(planes, planes)
        self.res2 = block(planes, planes)
        self.res3 = block(planes, planes)
        self.res4 = block(planes, planes)
        self.res5 = block(planes, planes)
        # 价值头
        self.conv2 = nn.Conv2d(16, 4, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4)
        self.fc1 = nn.Linear(100, 32)
        self.fc2 = nn.Linear(32, 1)
        # 策略头
        self.conv3 = nn.Conv2d(16, 2, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(2)
        self.fc3 = nn.Linear(50, 44)

    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        out = self.res5(out)
        # 价值头
        value_out = self.conv2(out)
        value_out = self.bn2(value_out)
        value_out = self.relu(value_out)
        value_out = value_out.view(-1, 100)
        value_out = self.fc1(value_out)
        value_out = F.tanh(self.fc2(value_out))
        # 策略头
        policy_out = self.conv3(out)
        policy_out = self.bn3(policy_out)
        policy_out = self.relu(policy_out)
        policy_out = policy_out.view(-1, 50)
        # policy_out = F.log_softmax(self.fc3(policy_out), dim=1)  # softmax+log pytorch 0.4 支持dim
        policy_out = F.log_softmax(self.fc3(policy_out))

        return policy_out,value_out

# device = torch.device("cpu")
# x = torch.Tensor(4,26,9,9).to(device)
# net = policy_value_net(BasicBlock,26,64).to(device)
# print(net)
# value_out,policy_out = net(x)
# print("value:",value_out.size())
# print("policy:",policy_out.size())


class PolicyValueNet(object):
    """策略价值网络 """
    def __init__(self,model_file=None, use_gpu=True):
        self.use_gpu = use_gpu
        self.l2_const = 1e-4  # 正则化系数
        if self.use_gpu:
            # device = torch.device("cuda:0")
            self.policy_value_net = policy_value_net(BasicBlock,14,16).cuda()
        else:
            # device = torch.device("cpu")
            self.policy_value_net = policy_value_net(BasicBlock,14,16)

        self.optimizer = optim.Adam(self.policy_value_net.parameters(), weight_decay=self.l2_const)

        if model_file:
            self.policy_value_net.load_state_dict(torch.load('ckpt/%s.pth'% model_file))

    def policy_value(self, state_batch):
        """
        输入：一批次的状态
        输出：一批次的落子概率和状态价值
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
        输入：棋盘
        输出：一个列表，由每一个可用落子的(action, probability)和棋盘状态价值组成
        """
        legal_positions = game.actions()  # 策略价值网络输出的是所有的落子概率，所以你需要剔除已落子的位置
        current_state = np.ascontiguousarray(game.state()).reshape([1,14,5,5])
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

        # 向前传播
        log_act_probs, value = self.policy_value_net(state_batch)
        # 损失公式： loss = (z - v)^2 - pi^T * log(p) + c||theta||^2，也就是value_loss+policy_loss
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1))
        loss = value_loss + policy_loss
        # 反向传播，优化损失
        loss.backward()
        self.optimizer.step()
        # 计算熵，只是用于监控
        entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, 1))
        return loss.data, entropy.data  # Change code for newest PyTorch version

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ 保存模型"""
        torch.save(self.policy_value_net.state_dict(), 'ckpt/%s.pth'%(model_file))


