# Train to create the model in ./model
# and predict to create the prediction in ./files
# and evaluate to create the evaluationg in ./results

# 1.use the last predict.py，it can complete the tasks:im2im、seq2seq
# 2.use the last triplet loss function, use the most similar neg to compute loss
# 3.use cosine similarity to compute loss and predicted features 
# 4.you can choose to use the GeMPooling instead of avgpooling in the end of resnet50 with "args.new_net=True"
import time
import torch
import predict
import evaluate
import argparse
import torchvision
from torch import nn
from pathlib import Path
from GeMPooling import GeMPooling 
from d2l import torch as d2l
from triplet_loss_cos import TripletLoss
from mapillary_sls.datasets.msls import MSLS
from mapillary_sls.utils.utils import configure_transform
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter


def get_net(new=False):
    """get the resnet50 and you can change something in it"""
    pretrained_net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    
    # if the pretrained_net is not good, use the net
    if new == False:
        net = pretrained_net
    else:
        """net = nn.Sequential(*list(pretrained_net.children())[:-2])
        net.add_module("gempooling", GeMPooling(pretrained_net.fc.in_features, output_size=(1, 1)))
        net.add_module("fc", pretrained_net.fc)"""
        net_list = list(pretrained_net.children())
        # create new net
        net = nn.Sequential()
        net.base = nn.Sequential(*net_list[:-2])
        # use an adaptiveavg-pooling in the GeMpooling,kernel_size=(1, 1)
        gem = GeMPooling(net_list[-1].in_features, output_size=(1, 1))
        net.back = nn.Sequential(gem, pretrained_net.fc)

    return net
    

def train(net, train_iter, num_epochs, loss, lr, optimizer, device, task, loss_log_path, seq_length, val_cities, predict_batch_size):
    """train funtion"""
    net.to(device)
    loss = loss
    optimizer=optimizer
    start_time = time.time()
    writer = SummaryWriter(loss_log_path)
    for epoch in range(num_epochs):
        net.train()
        metric = d2l.Accumulator(2)
        epoch_start = time.time()
        print(f'\n\nepoch{epoch + 1} is start:')
        for i, (sequences, labels) in enumerate(train_iter):
            if epoch == 0:
                N = labels.shape[1]
                q_seq_length, db_seq_length = split_seq(sequences, N, task)
            # sequences.shape=(batch_size, len(q)+len(p)+len(neg), 3, 480, 640)
            X = sequences.reshape(-1, 3, 480, 640).to(device)

            y_hat = net(X)
            y_hat = y_hat.reshape(sequences.shape[0], sequences.shape[1], -1)
            
            optimizer.zero_grad()
            
            l = loss(y_hat, q_seq_length, db_seq_length, N)
            
            l.backward()
            optimizer.step()
            metric.add(l * sequences.shape[0], labels.numel())
            
            train_loss = metric[0] / metric[1]
            
            if i % 1000 == 0:
                print(f'epoch:{epoch + 1}, batch:{i + 1}, loss:{train_loss:f}')
                niter = epoch * len(train_iter) + i
                writer.add_scalars("Train loss", {"train loss:": l.data.item()}, niter)
        print(f'epoch{epoch + 1} is end')
    
        # save the model trained
        model_path = f"./model/task{task}_epoch{epoch + 1}_lr{lr}all.params"
        torch.save(net.state_dict(), model_path)
        print(f'save the net successfully!!')

        # predict and save the keys
        print(f'\nStart to predict the keys of val cities: {val_cities}')
        predict_path = Path(f"./files/prediction_{task}_val_epoch{epoch + 1}lr{lr}.csv")
        predict.main(net, device, task, seq_length, predict_path, val_cities, predict_batch_size)
        print(f'save the prediction successfully!!')

        # evaluate the predictions above and save the results
        print(f'\nStart to evaluate the prediction of val cities: {val_cities}')
        evaluate_path = Path(f"./results/evaluate_task{task}_epoch{epoch + 1}lr{lr}.csv")
        evaluate.main(predict_path, evaluate_path, val_cities, task, seq_length)
        print(f'evaluate the model sucessfully! you can see the result in ./results/')

        epoch_end = time.time()
        print(f'\nnow loss:{train_loss:f}, time:{epoch_end - epoch_start}s({((epoch_end - epoch_start) / 60):.2} min)\n')
        print("****************************************************************")
    
    end_time = time.time()
    all_time = end_time - start_time

    writer.close()

    print(f'\n******the train is end *****************************************')
    print(f'in the end, cost time:{all_time // 60}min, loss:{train_loss:f}')
                       
            
def split_seq(sequences, N, task):
    """split the sequences before training according to the task"""
    if task == "im2im":
        q_seq_length, db_seq_length = 1, 1
    elif task == "seq2seq":
        seq_length = sequences.shape[1] // (N)
        q_seq_length, db_seq_length = seq_length, seq_length
    elif task == "im2seq":
        seq_length = (sequences.shape[1] - 1) // (N - 1)
        q_seq_length, db_seq_length = 1, seq_length
    elif task == "seq2im":
        seq_length = sequences.shape[1] - (N - 1)
        q_seq_length, db_seq_length = seq_length, 1
    
    return q_seq_length, db_seq_length


def create_dataloader(root_dir, cities, task, seq_length, batch_size):
    """load the trainDataset"""

    # get transform
    meta = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    transform = configure_transform(image_dim = (480, 640), meta = meta)

    # number of cached queries
    cached_queries = 60000
    # number of cached negatives
    cached_negatives = 80000

    # whether to use positive sampling
    positive_sampling = True
    
    # return the train dataset
    train_dataset = MSLS(root_dir, cities = cities, transform = transform, mode = 'train', 
                        task = task, seq_length = seq_length,negDistThr = 25, 
                        posDistThr = 5, nNeg = 5, cached_queries = cached_queries, 
                        cached_negatives = cached_negatives, positive_sampling = positive_sampling)
                    
    
    # divides dataset into smaller cache sets
    train_dataset.new_epoch()

    # creates triplets on the smaller cache set
    train_dataset.update_subcache()

    # create data loader
    # there are at least 7 images for batch_size=1(im2im)
    opt = {'batch_size': batch_size, 'shuffle': True}
    trainDataloader = DataLoader(train_dataset, **opt)
    
    return trainDataloader
    

def main():
    parser = argparse.ArgumentParser()

    # the location of msls_dataset in computer
    root_dir = Path('/datasets/msls').absolute()
    parser.add_argument('--task',
                        type=str,
                        default="im2im",
                        help='Task to train on:'
                             '[im2im, seq2im, im2seq, seq2seq]')
    parser.add_argument('--seq-length',
                        type=int,
                        default=1,
                        help='Sequence length to train for seq2X and X2seq tasks')
    parser.add_argument('--cuda',
                        type=str,
                        default="cuda:7",
                        help='Choose the gpu to use (cuda:*)|default=cuda:7')
    parser.add_argument('--loss-path',
                        type=Path,
                        default='./train_loss_log',
                        help='Path to loss log for tensorboardX.SummaryWriter')
    parser.add_argument('--num-epochs',
                        type=int,
                        default=30,
                        help='Set the number of epochs')
    parser.add_argument('--batch-size',
                        type=int,
                        default=5,
                        help='The size of each batch')
    parser.add_argument('--lr',
                        type=float,
                        default=0.0001,
                        help='Learning rate')
    parser.add_argument('--new-net',
                        type=bool,
                        default=False,
                        help='Choose to use the origin resnet50 or not')
    parser.add_argument('--predict-batch-size',
                        type=int,
                        default=190,
                        help='The size of each batch')
    parser.add_argument('--msls-root',
                        type=Path,
                        default=root_dir,
                        help='Path to the dataset')
    parser.add_argument('--cities',
                        type=str,
                        default="trondheim,london,boston,melbourne,amsterdam,helsinki,tokyo,toronto,saopaulo,moscow,zurich,paris,bangkok,budapest,austin,berlin,ottawa,phoenix,goa,amman,nairobi,manila",
                        help='The cities to train on')
    parser.add_argument('--val-cities',
                        type=str,
                        default='cph,sf',
                        help='Choose the cities to be predicted and evaluated')
    args = parser.parse_args()

    # create the train dataset first   (root_dir, cities, task, seq_length, batch_size)
    trainDataloader = create_dataloader(args.msls_root, args.cities, args.task, args.seq_length, args.batch_size)

    print("\n***************Load the trainDataset sucessfully**************")
    
    # get the net(new=True)you can choose to change something in the end
    net = get_net(new=args.new_net)
    
    print("***************Load the resnet sucessfully*********************")

    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    # loss = nn.TripletMarginLoss(margin=0.1, p=2)
    loss = TripletLoss(margin=0.1)
    # optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    optimizer = torch.optim.Adam([{"params":net.back.parameters()}, 
                                  {"params":net.base.parameters(), "lr":args.lr * 0.1}], 
                                  lr=args.lr)

    print("*************we will start training***************************")

    train(net, trainDataloader, args.num_epochs, loss, args.lr, optimizer, device, args.task, args.loss_path, args.seq_length, args.val_cities, args.predict_batch_size)


if __name__ == "__main__":
    main()