import torch
import torch.nn as nn
from torchtyping import TensorType
import math
import torch.nn.functional as F
class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.projection_Key = nn.Linear(embedding_dim, attention_dim, bias= False)
        self.projection_Query = nn.Linear(embedding_dim, attention_dim, bias= False)
        self.projection_Value = nn.Linear(embedding_dim, attention_dim, bias= False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        K = self.projection_Key(embedded)
        Q = self.projection_Query(embedded)
        V = self.projection_Value(embedded)
        attention_scores = Q @ K.permute(0,2,1) / math.sqrt(attention_dim)
        tril = torch.tril(torch.ones(attention_scores.shape[1], attention_scores.shape[1]))
        mask = (tril == 0)
        result = attention_scores.masked_fill(mask, float('-inf'))
        output = F.softmax(result, dim= 2)
        return torch.round(output @ V, decimals= 4)