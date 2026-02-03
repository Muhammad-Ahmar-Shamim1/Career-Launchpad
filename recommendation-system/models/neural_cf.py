import torch
import torch.nn as nn

class NeuralCF(nn.Module):
    def __init__(self, num_users, num_items):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, 50)
        self.item_embed = nn.Embedding(num_items, 50)
        self.fc = nn.Linear(100, 1)

    def forward(self, user, item):
        user_vec = self.user_embed(user)
        item_vec = self.item_embed(item)
        x = torch.cat([user_vec, item_vec], dim=1)
        return self.fc(x)
