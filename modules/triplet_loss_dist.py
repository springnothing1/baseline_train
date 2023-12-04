# use the L2 distance to find the hardest negtive
import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    '''
    Compute normal triplet loss or soft margin triplet loss given triplets
    '''
    def __init__(self, margin=0.1):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.loss = nn.MarginRankingLoss(margin=margin, reduction='mean')

    def forward(self, input, q_length, db_length, N):
        # num of samples in each batch
        n = input.shape[1]
        # compute the L2 distance between all the samples 
        dist = torch.pow(input, 2).sum(dim=2, keepdim=True).expand(input.shape[0], n, n)
        dist = dist + dist.transpose(1, 2)
        dist = dist - 2 * (torch.matmul(input, input.transpose(1, 2)))

        dist = dist[:, :q_length, q_length:].clamp(min=1e-12).sqrt()

        # compute the dist_ap and hardest(minim) dist_an
        dist_ap = dist[:, :, :db_length].mean([1, 2])
        dist_ans_split = [dist[:, :, (i + 1) * db_length:(i + 2) * db_length].mean([1, 2])\
                     for i in range(N - 2)]
        dist_ans = torch.stack(dist_ans_split, dim=1)
        # find the hardest negtive
        dist_an, _ = torch.min(dist_ans, dim=1)
        
        y = torch.ones_like(dist_an)
        loss = self.loss(dist_an, dist_ap, y)

        return loss


if __name__ == '__main__':
    loss = TripletLoss(0.2)
    input = torch.randn(size=(4, 7, 16))
    
    l = loss(input, 1, 1, 7)
    print(l)