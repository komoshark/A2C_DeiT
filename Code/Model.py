import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils import modify_DeiT, process_pre_states

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class A2C_DeiT(nn.Module):
    
    def __init__(self, historic_num, num_actions):
        super(A2C_DeiT, self).__init__()
        self.DeiT = modify_DeiT()
        #summary(self.DeiT, (3, 224, 224)) 
        self.feature_per_pic = 1
        self.reshape_w = 8
        self.reshape_h = 8
        self.historic_num = historic_num
        self.lin_num = (self.reshape_w / self.feature_per_pic) * (self.reshape_h / self.feature_per_pic)
        #self.maxpool = nn.MaxPool2d(kernel_size = self.feature_per_pic)
        input_num = 64
        #print('input_num',input_num)
        self.critic_linear = nn.Linear(input_num, 1) 
        self.actor_linear = nn.Linear(input_num, num_actions)

    def forward(self, states):
        outputs = torch.empty(0).to(device)
        out = torch.empty(0).to(device)
        imgs = process_pre_states(states).to(device)
        #print('imgs shape', imgs.shape)
        outputs = self.DeiT(imgs)
        outputs = outputs.view(outputs.size(0), -1)
        value = self.critic_linear(outputs)
        policy_dist = F.softmax(self.actor_linear(outputs), dim=-1)
        return value, policy_dist
    
class Simple_A2C(nn.Module): # an actor-critic neural network
    def __init__(self):
        super(Simple_A2C, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.gru = nn.GRUCell(32 * 16 * 16, 256)
        self.critic_linear, self.actor_linear = nn.Linear(256, 1), nn.Linear(256, 4)
    def forward(self, inputs, train=True, hard=False):
        x,hx = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        print(x.shape)
        hx = self.gru(x.view(-1, 32 * 16 * 16), (hx))
        return self.critic_linear(hx), self.actor_linear(hx), hx