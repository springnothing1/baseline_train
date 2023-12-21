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
import clip
import argparse
import clipvpr
import torchvision
from torch import nn
from pathlib import Path
from d2l import torch as d2l
from modules.utils import save_evaluate, reload_checkpoint, save_checkpoint, split_seq
from modules.loss import InfoNCELoss,TripletLoss, MultiSimilarityLoss
from modules.ResViT import ResTransformer
from modules.GeMPooling import GeMPooling 
from mapillary_sls.datasets.msls import MSLS
from mapillary_sls.datasets.msls_clip import MSLSCLIP
from mapillary_sls.utils.utils import configure_transform, clip_transform
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
        net, _ = clipvpr.load(clip_name="ViT-B/16", llama_name="BIAS-7B", llama_dir='./path/to/LLaMA/', llama_type="7B", 
        llama_download_root='ckpts', max_seq_len=512, phase="finetune", 
        prompt=['Is the scene in the picture urban or rural? How many lanes are there on the road in the photo? Is there a residential building in the picture? If so, which side of the road is it located on? Are there vegetation and trees in the photo? If so, which side of the road is it located on?'])
    
    elif net_name == "clip":
        net, _= clip.load("ViT-B/16")

    return net


def train_epoch(args, epoch, net, train_iter, optimizer, loss, writer, model_path):
    metric = d2l.Accumulator(2)
    global RECALL_BEST
    
    for i, (sequences, texts) in enumerate(train_iter):
        net.train()
        N = 2 + args.num_negatives
        q_seq_length, db_seq_length = split_seq(sequences, N, args.task)
        # sequences.shape=(batch_size, len(q)+len(p)+len(neg), 3, 480, 640)
        s_shape = sequences.shape
        images = sequences.reshape(-1, s_shape[-3], s_shape[-2], s_shape[-1]).to(next(net.parameters()).device)
        texts = texts.reshape(-1, texts.shape[-1]).to(next(net.parameters()).device)
        y_hat = net(images, texts)
        y_hat = y_hat.reshape(s_shape[0], s_shape[1], -1)
        
        optimizer.zero_grad()
        
        l = loss(y_hat, q_seq_length, db_seq_length, N)
        
        l.backward()
        optimizer.step()
        metric.add(l * sequences.shape[0], N)
        
        train_loss = metric[0] / metric[1]
        
        if (i % 1000 == 0) and (i != 0):
            print(f'epoch:[{epoch + 1}/{args.num_epochs}],\tbatch:[{i}/{len(train_iter)}],\tloss:{train_loss:f}')
            niter = epoch * len(train_iter) + i
            writer.add_scalars("Train loss", {"train loss:": l.data.item()}, niter)

        if (i % 10000 == 0) and (i != 0):
            # evaluate on val_cities
            recall_candidate = save_evaluate(args, net, epoch, i, cities='cph,sf')
            if recall_candidate > RECALL_BEST:
                # save the best reall@1 in one epoch
                RECALL_BEST = recall_candidate
                # set checkpoint
                save_checkpoint(net, optimizer, epoch, loss, model_path)
                print(f'\n++++save the best net with recall@1:{RECALL_BEST:.3} successfully!!\n')

    print(f"epoch{epoch + 1} if end ")           
    recall_candidate = save_evaluate(args, net, epoch, i, cities='cph,sf')
    if recall_candidate > RECALL_BEST:
        # save the best reall@1 in one epoch
        RECALL_BEST = recall_candidate
        # set checkpoint
        save_checkpoint(net, optimizer, epoch, loss, model_path)
        print(f'\n++++save the best net with recall@1:{RECALL_BEST:.3} successfully!!\n')
    return train_loss


def train(args, net, train_iter, loss, optimizer):
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
        train_loss = train_epoch(args, epoch, net, train_iter, optimizer, loss, writer, model_path)
            
        # save the models and evaluate on train_cities
        _ = save_evaluate(args, net, epoch, cities='trondheim,london,boston')

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


def create_dataloader(args):
    """load the trainDataset"""

    """# get transform
    meta = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    transform = configure_transform(image_dim = image_dim, meta = meta)"""
    
    if args.net_name in ["resnet50+gem", "resvit"]:
        image_dim = (480, 640)
        # get transform
        meta = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
        transform = configure_transform(image_dim = image_dim, meta = meta)
        

    elif args.net_name in ["vit", "clip","clipvpr"]:
        image_dim = (224, 224)
        transform = clip_transform(image_dim)

    # whether to use positive sampling
    positive_sampling = True
    
    # return the train dataset
    train_dataset = MSLSCLIP(root_dir=args.msls_root, cities = args.cities, transform = transform, mode = 'train', 
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
                        help='choose loss function: triplet or infonce or msloss')
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
    
    #optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.001, betas=(0.9,0.999), eps=1e-08)
    if net_name == "clipvpr":
        optimizer = torch.optim.AdamW(filter(lambda p : p.requires_grad, net.parameters()), 
                                      lr=args.lr, weight_decay=0.001, betas=(0.9,0.999), eps=1e-08)
    else:
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.001, betas=(0.9,0.999), eps=1e-08)
    
    print("\nloading.......\n")
    # create the train dataset first   (root_dir, cities, task, seq_length, batch_size)
    trainDataloader = create_dataloader(args)

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
    train(args, net, trainDataloader, loss, optimizer)


if __name__ == "__main__":
    main()