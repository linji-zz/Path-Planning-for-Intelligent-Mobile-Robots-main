import torch
import torch.optim as optim
import random
import numpy as np
from collections import deque
from models import TransformerDQN

class DQNAgent:
    def __init__(self, state_dim, seq_len, action_dim=8, lr=0.001, gamma=0.9, epsilon=0.9, memory_capacity=10000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.policy_net = TransformerDQN(input_dim=state_dim, seq_len=seq_len, output_dim=action_dim).to(self.device)
        self.target_net = TransformerDQN(input_dim=state_dim, seq_len=seq_len, output_dim=action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = deque(maxlen=memory_capacity)
        self.batch_size = 64
        self.seq_len = seq_len

    def select_action(self, state_seq):
        if random.random() < self.epsilon:
            # 随机选择一个子目标偏移量 (dx, dy) 的范围 -3~3
            return random.randint(0, self.action_dim - 1)  # 仍然离散，但代表方向
        with torch.no_grad():
            state_tensor = torch.FloatTensor(np.array(state_seq)).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def store_transition(self, state_seq, action, reward, next_state_seq, done):
        self.memory.append((state_seq, action, reward, next_state_seq, done))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)

        current_q = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = torch.nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self, episode, total_episodes):
        self.epsilon = max(0.01, 0.9 - episode / total_episodes * 0.89)