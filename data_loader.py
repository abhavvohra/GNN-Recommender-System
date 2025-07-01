import os
import pandas as pd
import numpy as np
import requests
import zipfile
from sklearn.preprocessing import LabelEncoder
import torch
from torch_geometric.data import Data

class MovieLensDataLoader:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
    def download_data(self):
        os.makedirs(self.data_dir, exist_ok=True)
        zip_path = os.path.join(self.data_dir, 'ml-100k.zip')
        
        if not os.path.exists(zip_path):
            print("Downloading MovieLens 100K dataset...")
            response = requests.get(self.url)
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            print("Dataset downloaded and extracted.")
    
    def load_ratings(self):
        self.download_data()
        ratings_path = os.path.join(self.data_dir, 'ml-100k', 'u.data')
        
        ratings = pd.read_csv(ratings_path, sep='\t', 
                            names=['user_id', 'item_id', 'rating', 'timestamp'])
        
        ratings['user_id'] = self.user_encoder.fit_transform(ratings['user_id'])
        ratings['item_id'] = self.item_encoder.fit_transform(ratings['item_id'])
        
        ratings['rating'] = ratings['rating'].astype(np.float32)
        
        return ratings
    
    def load_user_features(self):
        users_path = os.path.join(self.data_dir, 'ml-100k', 'u.user')
        
        users = pd.read_csv(users_path, sep='|', 
                          names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])
        
        users['user_id'] = self.user_encoder.transform(users['user_id'])
        users['gender'] = (users['gender'] == 'M').astype(int)
        
        occupation_encoder = LabelEncoder()
        users['occupation'] = occupation_encoder.fit_transform(users['occupation'])
        
        user_features = users[['age', 'gender', 'occupation']].values.astype(np.float32)
        user_features[:, 0] = (user_features[:, 0] - user_features[:, 0].mean()) / user_features[:, 0].std()
        
        return torch.FloatTensor(user_features)
    
    def load_item_features(self):
        items_path = os.path.join(self.data_dir, 'ml-100k', 'u.item')
        
        items = pd.read_csv(items_path, sep='|', encoding='latin-1',
                          names=['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + 
                                [f'genre_{i}' for i in range(19)])
        
        items['movie_id'] = self.item_encoder.transform(items['movie_id'])
        
        genre_features = items[[f'genre_{i}' for i in range(19)]].values.astype(np.float32)
        
        return torch.FloatTensor(genre_features)
    
    def create_bipartite_graph(self, ratings_df):
        num_users = len(self.user_encoder.classes_)
        num_items = len(self.item_encoder.classes_)
        
        edge_index_user_to_item = torch.stack([
            torch.LongTensor(ratings_df['user_id'].values),
            torch.LongTensor(ratings_df['item_id'].values + num_users)
        ])
        
        edge_index_item_to_user = torch.stack([
            torch.LongTensor(ratings_df['item_id'].values + num_users),
            torch.LongTensor(ratings_df['user_id'].values)
        ])
        
        edge_index = torch.cat([edge_index_user_to_item, edge_index_item_to_user], dim=1)
        
        edge_weight = torch.cat([
            torch.FloatTensor(ratings_df['rating'].values),
            torch.FloatTensor(ratings_df['rating'].values)
        ])
        
        user_features = self.load_user_features()
        item_features = self.load_item_features()
        
        # Make feature dimensions compatible
        feature_dim = max(user_features.shape[1], item_features.shape[1])
        
        # Pad user features to match max dimension
        if user_features.shape[1] < feature_dim:
            padding = torch.zeros(user_features.shape[0], feature_dim - user_features.shape[1])
            user_features = torch.cat([user_features, padding], dim=1)
        
        # Pad item features to match max dimension
        if item_features.shape[1] < feature_dim:
            padding = torch.zeros(item_features.shape[0], feature_dim - item_features.shape[1])
            item_features = torch.cat([item_features, padding], dim=1)
        
        x = torch.cat([user_features, item_features], dim=0)
        
        y = torch.FloatTensor(ratings_df['rating'].values)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y), num_users, num_items