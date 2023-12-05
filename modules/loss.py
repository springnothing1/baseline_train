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
        
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(input.device)
        loss = self.loss(logits / self.t, labels)
        
        return loss



class MultiSimilarityLoss(nn.Module):
    def __init__(self, thresh=0.5, margin=0.1, scale_pos=2.0, scale_neg=40.0):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = thresh
        self.margin = margin

        self.scale_pos = scale_pos
        self.scale_neg = scale_neg

    def forward(self, feats, q_length, *args):
        #assert feats.size(0) == labels.size(0), \
        #    f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        sim_mats = torch.matmul(feats, feats.transpose(1, 2))

        epsilon = 1e-5
        loss = list()

        for sim_mat in sim_mats:

            # get all positive pair simply by matching ground truth labels of those embedding which share the same label with anchor
            pos_pair_ = sim_mat[0][1:2]
            # remove the pair which calculates similarity of anchor with itself i.e the pair with similarity one.
            #pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]

            neg_pair_ = sim_mat[0][2:]

            # mine hard negatives using the method described in the blog, a margin of 0.1 is added to the neg pair similarity to fetch negatives which are just lying on the brink of boundary for hard negative which would have been missed if this term was not present
            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]
            pos_pair = pos_pair_[pos_pair_ - self.margin < max(neg_pair_)]

            print(len(neg_pair))
            print(len(pos_pair))
            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue

            # weighting step
            pos_loss = 1.0 / self.scale_pos * torch.log(
                1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
            neg_loss = 1.0 / self.scale_neg * torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))
            loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True)

        loss = sum(loss)# / batch_size
        return loss


# Triplet loss with cosine similarity
class TripletLoss(nn.Module):
    '''
    Compute normal triplet loss or soft margin triplet loss given triplets
    '''
    def __init__(self, margin=0.1):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
        self.loss = nn.MarginRankingLoss(margin=margin, reduction='mean')

    def forward(self, input, q_length, db_length, N, *args):
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



# Trplet loss with L2 distance
'''class TripletLoss(nn.Module):
    
    #Compute normal triplet loss or soft margin triplet loss given triplets
    
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

        return loss'''


if __name__ == "__main__":

    # InfoNCE loss:
    loss = InfoNCELoss()
    a = torch.randn(size=(3, 7, 30))
    q_length, db_seq_length = 1, 5
    l = loss(a, q_length)

    # Multi similarity loss:
    x = torch.randn(size=(3, 5, 16))
    #labels = torch.tensor([1, 2, 1, 4, 2])
    loss = MultiSimilarityLoss()
    l = loss(x, 1)
    print(l)

    # Triplet loss 
    loss = TripletLoss(0.2)
    input = torch.randn(size=(4, 7, 16))
    
    l = loss(input, 1, 1, 7)
    print(l)

