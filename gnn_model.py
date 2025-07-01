import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, MessagePassing
from torch_geometric.utils import add_self_loops, degree

class BipartiteGCN(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(BipartiteGCN, self).__init__(aggr='mean')
        self.lin = nn.Linear(in_channels, out_channels)
        
    def forward(self, x, edge_index, edge_weight=None):
        # Add self-loops and adjust edge weights accordingly
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=x.size(0))
        
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        return self.propagate(edge_index, x=x, norm=norm, edge_weight=edge_weight)
    
    def message(self, x_j, norm, edge_weight):
        if edge_weight is not None:
            return norm.view(-1, 1) * x_j * edge_weight.view(-1, 1)
        return norm.view(-1, 1) * x_j
    
    def update(self, aggr_out):
        return self.lin(aggr_out)

class GNNRecommender(nn.Module):
    def __init__(self, num_users, num_items, feature_dim=19, embedding_dim=64, hidden_dim=128, num_layers=3):
        super(GNNRecommender, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        self.feature_transform = nn.Linear(feature_dim, embedding_dim)
        
        self.gnn_layers = nn.ModuleList([
            BipartiteGCN(embedding_dim, hidden_dim)
        ])
        
        for _ in range(num_layers - 1):
            self.gnn_layers.append(BipartiteGCN(hidden_dim, hidden_dim))
        
        self.attention = nn.MultiheadAttention(hidden_dim, 8, batch_first=True)
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.rating_transform = nn.Linear(1, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        for layer in self.gnn_layers:
            nn.init.xavier_uniform_(layer.lin.weight)
    
    def forward(self, data, user_indices, item_indices):
        x = self.feature_transform(data.x)
        
        for gnn_layer in self.gnn_layers:
            x = F.relu(gnn_layer(x, data.edge_index, data.edge_attr))
            x = F.dropout(x, training=self.training)
        
        user_embeddings = x[:self.num_users]
        item_embeddings = x[self.num_users:]
        
        user_emb = user_embeddings[user_indices]
        item_emb = item_embeddings[item_indices]
        
        user_attn, _ = self.attention(user_emb.unsqueeze(1), 
                                    user_embeddings.unsqueeze(0).repeat(user_emb.size(0), 1, 1),
                                    user_embeddings.unsqueeze(0).repeat(user_emb.size(0), 1, 1))
        user_attn = user_attn.squeeze(1)
        
        item_attn, _ = self.attention(item_emb.unsqueeze(1),
                                    item_embeddings.unsqueeze(0).repeat(item_emb.size(0), 1, 1),
                                    item_embeddings.unsqueeze(0).repeat(item_emb.size(0), 1, 1))
        item_attn = item_attn.squeeze(1)
        
        combined = torch.cat([user_attn, item_attn], dim=1)
        
        rating_pred = self.predictor(combined)
        rating_pred = self.rating_transform(rating_pred) * 4 + 1
        
        return rating_pred.squeeze()

class CollaborativeFilteringLoss(nn.Module):
    def __init__(self, lambda_reg=0.01):
        super(CollaborativeFilteringLoss, self).__init__()
        self.lambda_reg = lambda_reg
        self.mse_loss = nn.MSELoss()
    
    def forward(self, predictions, targets, user_embeddings, item_embeddings):
        mse = self.mse_loss(predictions, targets)
        
        user_reg = torch.norm(user_embeddings) ** 2
        item_reg = torch.norm(item_embeddings) ** 2
        reg_loss = self.lambda_reg * (user_reg + item_reg) / (user_embeddings.size(0) + item_embeddings.size(0))
        
        return mse + reg_loss