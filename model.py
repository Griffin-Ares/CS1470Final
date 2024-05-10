import pickle
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
#from models.ast_models import ASTModel
from torch import Tensor
from models.custom_transformer import CustomTransformerModel, load_custom_weights
from models.AlignmentLayer import AlignmentLayer
#from models.im
import torch.nn.functional as F
import numpy as np


class Whisk(nn.Module):
    
    def __init__(self, embeddings_dim=768, nlayers=12, dropout=0.1, num_heads=12, trained=False, pretrained_model_path="pretrained_ast/audioset_10_10_0.4593.pth"):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.embeddings_dim = embeddings_dim
       # self.mean = torch.Tensor(np.load('data/global_mean.npy'))
       # self.std = torch.Tensor(np.load('data/global_std.npy'))
        
        # Load Pretrained AST Weights
        self.transformer = CustomTransformerModel(num_blocks=12, d_model=embeddings_dim, n_heads=12, dim_feedforward=embeddings_dim*4)

        if not trained:
            load_custom_weights(self.transformer, 'pretrained_ast/audioset_10_10_0.4593.pth')

        self.transformer.to(self.device)
        
        self.class_token = nn.Parameter(
            torch.randn(1, 1, embeddings_dim)
        )

      #  self.pos_enc = PositionalEncoding(embeddings_dim, dropout=dropout, max_len=1000)
      #  self.lin_out0 = nn.Linear(embeddings_dim, 512)

        
     #   self.dense1 = nn.Linear(768, 256) # First dimensionality reduction
        self.batch_norm = nn.BatchNorm1d(256)
       # self.dense2 = nn.Linear(256, 128)
        
        self.layernorm = nn.LayerNorm(embeddings_dim,eps=1e-12)
    #    self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.GELU()

      #  self.dropout = nn.Dropout(dropout)
        self.alignment = AlignmentLayer(768, 128, dropout=dropout)
    

    def normalize(self, x):
        """
        Normalize each embedding in a batch of embeddings to have a magnitude (L2 norm) of 1.
        
        Parameters:
        - x: A PyTorch tensor of shape (batch_size, embedding_dim) to be normalized.
        
        Returns:
        - A tensor where each embedding is normalized to a magnitude of 1.
        """
        # Calculate the L2 norm of each embedding in the batch.
        # The 'dim=1' parameter computes the norm across the embedding dimension for each sample in the batch.
       # l2_norms = torch.norm(x, p=2, dim=1, keepdim=True)
       # # Normalize each embedding in the batch to have a magnitude of 1.
       # normalized_tensor = x / l2_norms
        return F.normalize(x, p=2, dim=-1)#normalized_tensor
    


    def forward(self, x) -> Tensor:
        batch_size = x.shape[0]
        num_patches = x.shape[1]
        x = x.to(self.device)
        

        x = self.transformer(x)
        x = self.alignment(x)
        
        return x

