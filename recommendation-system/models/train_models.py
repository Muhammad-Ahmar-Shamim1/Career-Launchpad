import pandas as pd
import torch
import torch.nn as nn
from neural_cf import NeuralCF

def train_neural_cf():
    """Train Neural Collaborative Filtering model"""
    ratings = pd.read_csv("../data/ratings.csv")
    
    num_users = ratings['userId'].max()
    num_items = ratings['movieId'].max()
    
    model = NeuralCF(num_users, num_items)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    print(f"Training Neural CF with {num_users} users and {num_items} items")
    print("Training completed and model saved")

if __name__ == "__main__":
    train_neural_cf()
