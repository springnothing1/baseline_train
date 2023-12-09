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
import clipvpr
import torchvision
from torch import nn
from pathlib import Path
from d2l import torch as d2l
from modules.loss import InfoNCELoss
from modules.loss import TripletLoss
from modules.loss import MultiSimilarityLoss
from modules.ResViT import ResTransformer
from modules.GeMPooling import GeMPooling 
from mapillary_sls.datasets.msls import MSLS
from mapillary_sls.utils.utils import configure_transform
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter


# save the best recall@1
RECALL_BEST = 0

def get_net(net_name = "resnet50+gem"):
    """get the net of resnet50 with gempooling or ViT_base_16 """
    
    if net_name == "resnet50+gem":
        pretrained_net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        net_list = list(pretrained_net.children())

        # create new net
        net = nn.Sequential()
        net.base = nn.Sequential(*net_list[:-2])

        # use an adaptiveavg-pooling in the GeMpooling,kernel_size=(1, 1)
        gem = GeMPooling(net_list[-1].in_features, output_size=(1, 1))
        net.back = nn.Sequential()
        net.back.add_module("gem", gem)
        net.back.add_module("fc", nn.Linear(in_features=2048, out_features=2048, bias=True))
        
        
    elif net_name == "vit":
        net = torchvision.models.vit_b_16(weights='IMAGENET1K_V1')
        # net.heads.head.out_features = 2048

    elif net_name == "resvit":
        net = ResTransformer()
        
    elif net_name == "clipvpr":
        net, process = clipvpr.load(clip_name="ViT-B/16", llama_name="BIAS-7B", llama_dir='./path/to/LLaMA/', llama_type="7B", 
        llama_download_root='ckpts', max_seq_len=512, phase="finetune", 
        prompt=['Is the scene in the picture urban or rural? How many lanes are there on the road in the photo? Is there a residential building in the picture? If so, which side of the road is it located on? Are there vegetation and trees in the photo? If so, which side of the road is it located on?'])

    return net


def save_evaluate(args, net, epoch, image_dim, i=999999, cities='cph,sf'):
    outpath = args.out_path

    # predict and save the keys
    print(f'\nStart to predict the keys of cities: {cities}')
    predict_path = Path(os.path.join(outpath, Path(f"prediction_{cities}_val_epoch{epoch + 1}_i{i}.csv")))
    predict.main(args, net, image_dim, predict_path, cities)
    print(f'save the prediction successfully!!')

    # evaluate the predictions above and save the results
    print(f'\nStart to evaluate the prediction of cities: {cities}')
    evaluate_path = Path(os.path.join(outpath, Path(f"evaluate_task{cities}_epoch{epoch + 1}_i{i}.csv")))
    recall_1 = evaluate.main(args, predict_path, evaluate_path, cities)
    print(f'evaluate the model sucessfully! you can see the result in {outpath}')
    return recall_1


def reload_checkpoint(net, optimizer, path_checkpoint):
    # load the checkpoint
    checkpoint = torch.load(path_checkpoint, map_location='cpu')

    start_epoch = checkpoint['epoch']
    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    loss = checkpoint['loss']

    return net, optimizer, start_epoch, loss


def save_checkpoint(net, optimizer, epoch, loss, model_path):
    checkpoint = {
        'epoch':epoch,
        'net':net.state_dict(),
        'optimizer':optimizer.state_dict(),
        'loss':loss}
    torch.save(checkpoint, model_path)


def train_epoch(args, epoch, net, train_iter, optimizer, loss, writer, image_dim, model_path):
    net.train()
    metric = d2l.Accumulator(2)
    global RECALL_BEST
    
    for i, (sequences, labels) in enumerate(train_iter):
        N = labels.shape[1]
        q_seq_length, db_seq_length = split_seq(sequences, N, args.task)
        # sequences.shape=(batch_size, len(q)+len(p)+len(neg), 3, 480, 640)
        s_shape = sequences.shape
        X = sequences.reshape(-1, s_shape[-3], s_shape[-2], s_shape[-1]).to(next(net.parameters()).device)

        y_hat = net(X)
        y_hat = y_hat.reshape(s_shape[0], s_shape[1], -1)
        
        optimizer.zero_grad()
        
        l = loss(y_hat, q_seq_length, db_seq_length)
        
        l.backward()
        optimizer.step()
        metric.add(l * sequences.shape[0], labels.numel())
        
        train_loss = metric[0] / metric[1]
        
        if (i % 1000 == 0) and (i != 0):
            print(f'epoch:[{epoch + 1}/{args.num_epochs}],\tbatch:[{i}/{len(train_iter)}],\tloss:{train_loss:f}')
            niter = epoch * len(train_iter) + i
            writer.add_scalars("Train loss", {"train loss:": l.data.item()}, niter)

        if (i % 10000 == 0) and (i != 0):
            # evaluate on val_cities
            recall_candidate = save_evaluate(args, net, epoch, image_dim, i, cities='cph,sf')
            if recall_candidate > RECALL_BEST:
                # save the best reall@1 in one epoch
                RECALL_BEST = recall_candidate
                # set checkpoint
                save_checkpoint(net, optimizer, epoch, loss, model_path)
                print(f'++++save the best net with recall@1:{RECALL_BEST:.3} successfully!!')

    print(f"epoch{epoch + 1} if end ")           
    recall_candidate = save_evaluate(args, net, epoch, image_dim, i, cities='cph,sf')
    if recall_candidate > RECALL_BEST:
        # save the best reall@1 in one epoch
        RECALL_BEST = recall_candidate
        # set checkpoint
        save_checkpoint(net, optimizer, epoch, loss, model_path)
        print(f'\n++++save the best net with recall@1:{RECALL_BEST:.3} successfully!!\n')
    return train_loss


def train(args, net, train_iter, loss, optimizer, image_dim):
    """train funtion"""
    start_time = time.time()
    writer = SummaryWriter(args.loss_path)
    start_epoch = -1

    # create the path to save the models and evaluate results
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    model_path = Path(os.path.join(args.out_path, Path("model_best.pt")))

    # reload the checkpoint if resume==True
    if args.resume:
        net, optimizer, start_epoch, loss = reload_checkpoint(net, optimizer, model_path)

    for epoch in range(start_epoch + 1, args.num_epochs):
        epoch_start = time.time()
        print(f'\n\nepoch [{epoch + 1}/{args.num_epochs}] is start:')
        
        # train for every epoch and get the best recall@1
        train_loss = train_epoch(args, epoch, net, train_iter, optimizer, loss, writer, image_dim, model_path)
            
        # save the models and evaluate on train_cities
        _ = save_evaluate(args, net, epoch, image_dim, cities='trondheim,london,boston')

        # record the time
        epoch_end = time.time()
        print(f'epoch{epoch} is end')
        print(f'now loss:{train_loss:f}, time:{((epoch_end - epoch_start) / 60):.3} min ({((epoch_end - epoch_start) / 3600):.3} hours)\n')
        print("****************************************************************")
    
    end_time = time.time()
    all_time = end_time - start_time

    writer.close()

    print(f'\n******the train is end *****************************************')
    print(f'in the end, cost time:{all_time // 3600}hours, loss:{train_loss:f}')
                       
            
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


def create_dataloader(args, transform):
    """load the trainDataset"""

    """# get transform
    meta = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    transform = configure_transform(image_dim = image_dim, meta = meta)"""

    # whether to use positive sampling
    positive_sampling = True
    
    # return the train dataset
    train_dataset = MSLS(root_dir=args.msls_root, cities = args.cities, transform = transform, mode = 'train', 
                        task = args.task, seq_length = args.seq_length, negDistThr = 25, 
                        posDistThr = 5, nNeg = args.num_negatives, cached_queries = args.cached_queries, 
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
    parser.add_argument('--num-negatives',
                        type=int,
                        default=5,
                        help='choose the number of negatives in one triplet')
    parser.add_argument('--loss',
                        type=str,
                        default="triplet",
                        help='choose loss function: triplet or infonce')
    parser.add_argument('--resume',
                        type=bool,
                        default=False,
                        help='if you want to train from the checkpoint')
    parser.add_argument('--predict-batch-size',
                        type=int,
                        default=512,
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
                        default=300000,
                        help='The length of cached queries')
    parser.add_argument('--cached-negatives',
                        type=int,
                        default=600000,
                        help='The length of cached queries')
    parser.add_argument('--val-cities',
                        type=str,
                        default='cph,sf',
                        help='Choose the cities to be predicted and evaluated')
    args = parser.parse_args()
    
    # print the configure of training
    print("the configure of training :")
    for k, v in args.__dict__.items():
        print(f"{k}: {v}")

    # get the net(new=True)you can choose to change something in the end
    net_name = args.net_name
    net = get_net(net_name)
    
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.001, betas=(0.9,0.999), eps=1e-08)
    
    if net_name in ["resnet50+gem", "resvit"]:
        image_dim = (480, 640)
        # get transform
        meta = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
        transform = configure_transform(image_dim = image_dim, meta = meta)
        

    elif net_name in ["vit", "clipvpr"]:
        image_dim = (224, 224)
        transform = clipvpr.transform(image_dim)
    
    print("\nloading.......\n")
    # create the train dataset first   (root_dir, cities, task, seq_length, batch_size)
    trainDataloader = create_dataloader(args, transform)

    # choose the device to train the net
    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")

    # choose the loss function used to train the net
    if args.loss == "triplet":
        loss = TripletLoss(margin=0.1)

    elif args.loss == "infonce":
        loss = InfoNCELoss(t=0.02)

    elif args.loss == "msloss":
        loss = MultiSimilarityLoss(thresh=0.5, margin=0.1, scale_pos=2.0, scale_neg=40.0)
    
    print("\n******************we will start training************************")

    net.to(device)
    train(args, net, trainDataloader, loss, optimizer, image_dim)


if __name__ == "__main__":
    main()