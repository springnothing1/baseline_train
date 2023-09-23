# Train to create the model in ./model
# and predict to create the prediction in ./files
# and evaluate to create the evaluationg in {outpath}

# 1.use the last predict.py，it can complete the tasks:im2im、seq2seq
# 2.use the last triplet loss function, use the most similar neg to compute loss
# 3.use cosine similarity to compute loss and predicted features 
# 4.you can choose to use the GeMPooling instead of avgpooling in the end of resnet50 with "args.new_net=True"
import os
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


def get_net(net_name = "resnet50+gem"):
    """get the net of resnet50 with gempooling or ViT_base_16 """
    
    if net_name == "resnet50+gem":
        pretrained_net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
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
    elif net_name == "vit":
        net = torchvision.models.vit_b_16(weights='IMAGENET1K_V1')
    return net


def save_evaluate(args, net, epoch, image_dim):
    task = args.task
    outpath = args.out_path

    # create the path to save the models and evaluate results
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # save the model trained
    model_path = Path(os.path.join(outpath, Path(f"task{task}_epoch{epoch + 1}_lr{args.lr}.params")))
    torch.save(net.state_dict(), model_path)
    print(f'save the net successfully!!')

    # predict and save the keys
    print(f'\nStart to predict the keys of val cities: {args.val_cities}')
    predict_path = Path(os.path.join(outpath, Path(f"prediction_{task}_val_epoch{epoch + 1}lr{args.lr}.csv")))
    predict.main(net, task, image_dim, args.seq_length, predict_path, args.val_cities, args.predict_batch_size)
    print(f'save the prediction successfully!!')

    # evaluate the predictions above and save the results
    print(f'\nStart to evaluate the prediction of val cities: {args.val_cities}')
    evaluate_path = Path(os.path.join(outpath, Path(f"evaluate_task{task}_epoch{epoch + 1}lr{args.lr}.csv")))
    evaluate.main(predict_path, evaluate_path, args.val_cities, task, args.seq_length)
    print(f'evaluate the model sucessfully! you can see the result in {outpath}')

def train(args, net, train_iter, loss, optimizer, device, image_dim):
    """train funtion"""
    
    net.to(device)
    start_time = time.time()
    writer = SummaryWriter(args.loss_path)

    for epoch in range(args.num_epochs):
        net.train()
        metric = d2l.Accumulator(2)
        epoch_start = time.time()
        print(f'\n\nepoch{epoch + 1} is start:')
        for i, (sequences, labels) in enumerate(train_iter):
            if epoch == 0:
                N = labels.shape[1]
                q_seq_length, db_seq_length = split_seq(sequences, N, args.task)
            # sequences.shape=(batch_size, len(q)+len(p)+len(neg), 3, 480, 640)
            s_shape = sequences.shape
            X = sequences.reshape(-1, s_shape[-3], s_shape[-2], s_shape[-1]).to(device)

            y_hat = net(X)
            y_hat = y_hat.reshape(s_shape[0], s_shape[1], -1)
            
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
    
        # save the models and evaluate the predictions from the net now
        save_evaluate(args, net, epoch, image_dim)

        # record the time
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


def create_dataloader(args, image_dim):
    """load the trainDataset"""

    # get transform
    meta = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    transform = configure_transform(image_dim = image_dim, meta = meta)

    # number of cached queries
    # cached_queries = 60000
    # number of cached negatives
    # cached_negatives = 80000

    # whether to use positive sampling
    positive_sampling = True
    
    # return the train dataset
    train_dataset = MSLS(root_dir=args.msls_root, cities = args.cities, transform = transform, mode = 'train', 
                        task = args.task, seq_length = args.seq_length, negDistThr = 25, 
                        posDistThr = 5, nNeg = 5, cached_queries = args.cached_queries, 
                        cached_negatives = args.cached_negatives, positive_sampling = positive_sampling)
                    
    
    # divides dataset into smaller cache sets
    train_dataset.new_epoch()

    # creates triplets on the smaller cache set
    train_dataset.update_subcache()

    # create data loader
    # there are at least 7 images for batch_size=1(im2im)
    opt = {'batch_size': args.batch_size, 'shuffle': True}
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
    parser.add_argument('--out-path',
                        type=Path,
                        default='./results',
                        help='Path of models&predictions&evaluates')
    parser.add_argument('--num-epochs',
                        type=int,
                        default=30,
                        help='Set the number of epochs')
    parser.add_argument('--batch-size',
                        type=int,
                        default=5,
                        help='The size of each batch:5 for im2im resnet;2 1 for seq2seq renet')
    parser.add_argument('--lr',
                        type=float,
                        default=0.00001,
                        help='Learning rate')
    parser.add_argument('--net-name',
                        type=str,
                        default="resnet50+gem",
                        help='choose net: resnet50+gem or vit')
    parser.add_argument('--predict-batch-size',
                        type=int,
                        default=190,
                        help='The size of predict batch,190 for resnet50,')
    parser.add_argument('--msls-root',
                        type=Path,
                        default=root_dir,
                        help='Path to the dataset')
    parser.add_argument('--cities',
                        type=str,
                        default="trondheim,london,boston,melbourne,amsterdam,helsinki,tokyo,toronto,saopaulo,moscow,zurich,paris,bangkok,budapest,austin,berlin,ottawa,phoenix,goa,amman,nairobi,manila",
                        help='The cities to train on')
    parser.add_argument('--cached-queries',
                        type=int,
                        default=200000,
                        help='The length of cached queries')
    parser.add_argument('--cached-negatives',
                        type=int,
                        default=400000,
                        help='The length of cached queries')
    parser.add_argument('--val-cities',
                        type=str,
                        default='cph,sf',
                        help='Choose the cities to be predicted and evaluated')
    args = parser.parse_args()
    
    # get the net(new=True)you can choose to change something in the end
    net_name = args.net_name
    net = get_net(net_name)
    
    if net_name == "resnet50+gem":
        optimizer = torch.optim.Adam([{"params":net.back.parameters(),  "lr":args.lr * 10}, 
                                    {"params":net.base.parameters()}], 
                                    lr=args.lr)
        image_dim = (480, 640)
    elif net_name == "vit":
        # optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        optimizer = torch.optim.Adam([{"params":net.conv_proj.parameters()}, 
                                      {"params":net.encoder.parameters()},
                                      {"params":net.heads.parameters(), "lr":args.lr * 100}], 
                                    lr=args.lr)
        image_dim = (224, 224)
    
    print(f"***************Load the {net_name} net sucessfully*********************")

    # create the train dataset first   (root_dir, cities, task, seq_length, batch_size)
    trainDataloader = create_dataloader(args, image_dim)

    print("\n***************Load the trainDataset sucessfully**************")

    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    # loss = nn.TripletMarginLoss(margin=0.1, p=2)
    loss = TripletLoss(margin=0.1)
    print("*************we will start training***************************")

    train(args, net, trainDataloader, loss, optimizer, device, image_dim)


if __name__ == "__main__":
    main()