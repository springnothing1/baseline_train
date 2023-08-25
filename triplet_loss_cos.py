# use the cos similarity to find the hardest negtive
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
        # input1 is the anchor, input2 contains pos and neg
        input1 = input[:, :q_length, :]
        input2 = input[:, q_length:, :]
        input1_normlized = input1 / torch.norm(input1, dim=2, keepdim=True)
        input2_normlized = input2 / torch.norm(input2, dim=2, keepdim=True)

        # compute thr cos similarity between anchor and pos/neg
        sim = torch.matmul(input1_normlized, input2_normlized.transpose(1, 2))

        # get the sim_ap and hardest(maxim) sim_an
        sim_ap = sim[:, :, :db_length].mean([1, 2])
        sim_an_split = [sim[:, :, (i + 1) * db_length:(i + 2) * db_length].mean([1, 2])\
                   for i in range(N - 2)]
        sim_ans = torch.stack(sim_an_split, dim=1)
        sim_an, _ = torch.max(sim_ans, dim=1)

        y = torch.ones_like(sim_an)

        loss = self.loss(sim_ap, sim_an, y)

        return loss
    

if __name__ == '__main__':
    loss = TripletLoss(0.2)
    input = torch.randn(size=(4, 7, 16))
    
    l = loss(input, 1, 1, 7)
    print(l)