import torch
import numpy as np
from torch import nn

class Attention(nn.Module):

    def __init__(self, encoder_dim: int, decoder_dim: int):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim

    def forward(self, 
                query: torch.Tensor,  # [decoder_dim]
                values: torch.Tensor, # [seq_length, encoder_dim]
    ):
        weights = self._get_weights(query, values) # [seq_length]
        weights = nn.functional.softmax(weights, dim=0)
        return weights @ values  # [encoder_dim]

class AdditiveAttention2(Attention):

    def __init__(self, encoder_dim, decoder_dim):
        super().__init__(encoder_dim, decoder_dim)
        self.v = nn.Parameter(
            torch.FloatTensor(self.decoder_dim).uniform_(-0.1, 0.1))
        self.W_1 = nn.Linear(self.decoder_dim, self.decoder_dim)
        self.W_2 = nn.Linear(self.encoder_dim, self.decoder_dim)

    def _get_weights(self,        
                     query: torch.Tensor,  # [decoder_dim]
                     values: torch.Tensor,  # [seq_length, encoder_dim]
    ):
        query = query.repeat(values.size(0), 1)  # [seq_length, decoder_dim]
        weights = self.W_1(query) + self.W_2(values)  # [seq_length, decoder_dim]
        return torch.tanh(weights) @ self.v  # [seq_length]

class MultiplicativeAttention(Attention):

    def __init__(self, encoder_dim: int, decoder_dim: int):
        super().__init__(encoder_dim, decoder_dim)
        self.W = nn.Parameter(torch.FloatTensor(
            self.decoder_dim, self.encoder_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.W)

    def _get_weights(self,
                     query: torch.Tensor,  # [decoder_dim]
                     values: torch.Tensor, # [seq_length, encoder_dim]
    ):
        weights = query.matmul(self.W).matmul(values.T)  # [seq_length]
        return weights  # /np.sqrt(self.decoder_dim)  # [seq_length]    

    
class AdditiveAttention(Attention):
    # try broadcast
    def __init__(self, encoder_dim, decoder_dim):
        super().__init__(encoder_dim, decoder_dim)
        self.v = nn.Parameter(
            torch.FloatTensor(self.decoder_dim).uniform_(-0.1, 0.1))
        self.W1 = nn.Parameter(torch.FloatTensor(
            self.decoder_dim, self.decoder_dim).uniform_(-0.1, 0.1))
        self.W2 = nn.Parameter(torch.FloatTensor(
            self.encoder_dim, self.decoder_dim).uniform_(-0.1, 0.1))
        

    def _get_weights(self,        
                     query: torch.Tensor,  # [decoder_dim]
                     values: torch.Tensor,  # [seq_length, encoder_dim]
    ):
        weights = torch.matmul(query.unsqueeze(1), self.W1) + torch.matmul(values, self.W2)  # [seq_length, decoder_dim]
        weights = torch.matmul(torch.tanh(weights), self.v).squeeze(-1)
        
        return weights  # [seq_length]    
