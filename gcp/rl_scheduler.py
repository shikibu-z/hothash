import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Query:
    """Represents a database query"""
    id: str  # Unique identifier for the query
    data_items: List[str]  # Data items this query needs
    op: str
    estimated_cost: float = 1.0  # Processing cost estimate
    priority: int = 1


class DQNNetwork(nn.Module):
    """Deep Q-Network for query scheduling"""

    def __init__(self,
                 state_size: int,
                 action_size: int,
                 hidden_size: int = 256):
        super(DQNNetwork, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class RLQueryScheduler:
    """Reinforcement Learning Query Scheduler"""

    def __init__(self, state_size, num_nodes):
        self.state_size = state_size
        self.action_size = num_nodes

        # Neural networks
        self.q_network = DQNNetwork(self.state_size, self.action_size)

    def act(self, state, valid_actions):
        """Choose action using the trained policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)

        # Mask invalid actions
        masked_q_values = q_values.clone()
        for i in range(self.action_size):
            if i not in valid_actions:
                masked_q_values[0][i] = float('-inf')

        return masked_q_values.argmax().item()

    def schedule_query(self, state, valid_actions) -> int:
        """Schedule a query using the trained policy"""
        action = self.act(state, valid_actions)
        return action

    def load_checkpoint(self, filepath="rl_scheduler_checkpoint_v2.pth"):
        """Load model, memory and training state from file"""
        if not os.path.exists(filepath):
            print(f"❌ Checkpoint file {filepath} not found.")
            return

        checkpoint = torch.load(filepath, weights_only=False)
        self.q_network.load_state_dict(checkpoint['q_network'])
        print(f"✅ Checkpoint loaded from {filepath}")

    def save_checkpoint(self, filepath="rl_scheduler_checkpoint_v2.pth"):
        """Save model, memory and training state to file"""
        print(f"Saving checkpoint to {filepath}...")
        checkpoint = {'q_network': self.q_network.state_dict()}
        torch.save(checkpoint, filepath)
        print(f"✅ Checkpoint saved to {filepath}")
