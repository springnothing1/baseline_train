
import torch
from torch import nn

#from ret_benchmark.losses.registry import LOSS


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


def main():
    x = torch.randn(size=(3, 5, 16))
    #labels = torch.tensor([1, 2, 1, 4, 2])
    loss = MultiSimilarityLoss()
    l = loss(x, 1)

    print(l)

if __name__ == "__main__":
    main()