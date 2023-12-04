import torch
from torch import nn


class InfoNCELoss(nn.Module):
    def __init__(self, t=0.02):
        super().__init__()
        self.t = t
        self.loss = nn.CrossEntropyLoss()
        
    def forward(self, input, q_length, *args):
        # input1 is the anchor, input2 contains pos and neg
        input1 = input[:, :q_length, :]
        input2 = input[:, q_length:, :]
        input1_normlized = input1 / torch.norm(input1, dim=2, keepdim=True)
        input2_normlized = input2 / torch.norm(input2, dim=2, keepdim=True)

        # compute thr cos similarity between anchor and pos/neg
        logits = torch.matmul(input1_normlized, input2_normlized.transpose(1, 2)).squeeze(1)
        
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(args[-1])
        loss = self.loss(logits / self.t, labels)
        
        return loss


if __name__ == "__main__":
    loss = InfoNCELoss()
    a = torch.randn(size=(3, 7, 30))
    q_length, db_seq_length, N, device = 1, 5, 7, torch.device("cpu")
    l = loss(a, q_length, db_seq_length, N, device)