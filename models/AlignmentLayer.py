import torch
import torch.nn as nn
import torch.nn.functional as F

class AlignmentLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.3):
        super().__init__()
        layer_size = 256#max(int(in_dim / 3), out_dim * 2)
       # self.dense1 = nn.Linear(in_dim, layer_size) # First dimensionality reduction
       # self.batch_norm = nn.BatchNorm1d(layer_size)
        self.dense2 = nn.Linear(768, 128)
        #self.dense2 = self.dense2.double()
       # self.dropout = nn.Dropout(dropout)

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
        # Normalize each embedding in the batch to have a magnitude of 1.
       # normalized_tensor = x / l2_norms
        return F.normalize(x, p=2, dim=-1)

    def forward(self, x):
        #x = x.double()
        x = self.dense2(x)
        x = self.normalize(x)
        return x
