import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from sac_discrete.memory import LazyMultiStepMemory, LazyPrioritizedMultiStepMemory



def initialize_weights_he(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 3, 3, 2),        
            nn.ReLU(),
            nn.Conv2d(3, 6, 3, 1),         
            nn.ReLU(),
            nn.Conv2d(6, 6, 3, 1),         
            nn.ReLU(),
            Flatten()
        ).apply(initialize_weights_he)

        self.fc1 = nn.Linear(args.n_states, args.hid_size)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(args.hid_size, args.n_actions)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.net(x)
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self, args):
        self.args = args
        self.eval_net, self.target_net = Net(args), Net(args)

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = LazyMultiStepMemory(
                                        capacity=args.memory_size,
                                        state_shape=(2, args.out_shape, args.out_shape),
                                        device=args.device, gamma=args.gamma, multi_step=args.multi_step)

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=args.lr)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < self.args.epsilon:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()[0]
        else:   # random
            action = np.random.randint(0, self.args.n_actions)
        return action

    def learn(self):
        # target parameter update
        if self.learn_step_counter % self.args.target_update == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        batch = self.memory.sample(self.args.batch_size)
        states, actions, rewards, next_states, dones = batch

        # q_eval w.r.t the action in experience
        
        q_eval = self.eval_net(states).gather(1, actions.long())  # shape (batch, 1)
        q_next = self.target_net(next_states).detach()     # detach from graph, don't backpropagate
        q_target = rewards + self.args.gamma * q_next.max(1)[0].view(self.args.batch_size, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # print ("Learn function is called!")

    def load_models(self, load_dir, level, episode_number):
        model_path = os.path.join(load_dir, level + '_policy_' + str(episode_number) + '.pth')
        model_name = os.path.join(level + '_policy_' + str(episode_number))
        print ("Model ", model_path, " is loaded!")
        self.eval_net.load_state_dict(torch.load(model_path))
        self.eval_net.eval()
        self.target_net.load_state_dict(torch.load(os.path.join(load_dir, level + '_target_net_' + str(episode_number) + '.pth')))
        self.target_net.eval()

        return model_name

    def save_models(self, save_dir, level, episode_number):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.eval_net.state_dict(), os.path.join(save_dir, level + '_policy_' + str(episode_number) + '.pth'))
        torch.save(self.target_net.state_dict(), os.path.join(save_dir, level + '_target_net_' + str(episode_number) + '.pth'))