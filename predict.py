import os
import torch
import argparse
import torchvision
from torch import nn
from pathlib import Path
from mapillary_sls.datasets.msls import MSLS
from mapillary_sls.datasets.generic_dataset import ImagesFromList
from mapillary_sls.utils.utils import configure_transform
from torch.utils.data import DataLoader


def predict_feature(net, Loader, device, im_or_seq='im'):
    """compute the features with net trained and get indices"""
    net.to(device)
    net.eval()
    result = []
    idx = []
    with torch.no_grad():
        if im_or_seq == 'im':

            for x, y in Loader:
                x = x.to(device)
                y_hat = net(x)
                result.append(y_hat)
                idx.append(y)
                
        elif im_or_seq == 'seq':
            # type(x_list)=list, and len(x_list=seq_length)
            for x_list, y in Loader:
                y_hat_list = torch.zeros((x_list[0].shape[0], net.back[1].out_features)).to(device)
                seq_length = len(x_list)
                for x in x_list:
                    # now the shape of x is(batch_size, 3, 224, 224)
                    x = x.to(device)
                    y_hat = net(x)
                    # compute the mean of all images in the seq
                    y_hat_list += y_hat
                y_hat = y_hat_list / seq_length
                result.append(y_hat)
                idx.append(y)  
                
    result = torch.cat(result, dim=0)
    idx = torch.cat(idx, dim=0).reshape(-1, 1)
    return result, idx


def query_to_dbIdx(qfeature, dbfeature):
    """find the first 20 most similar db_index to each query"""
    # for L2 norm:
    qfeature_normed = qfeature / torch.norm(qfeature, dim=1, keepdim=True)
    dbfeature_normed = dbfeature / torch.norm(dbfeature, dim=1, keepdim=True)

    # cos<a,b> of two vecter: a·b / (|a|*|b|)  == similarity of two vector
    sim = torch.matmul(qfeature_normed, dbfeature_normed.transpose(0, 1))
    # get the index of the first 20 maximum in sim
    idx = torch.argsort(sim, dim=1, descending=True)[:, :20]
    
    return idx


def find_keys(indices, images):
    """find the image keys according to indices"""
    address_all = images[indices]
    # save the keys of all the queries
    keys_all = []
    for address_query in address_all:
        
        keys = []
        for address in address_query:
            # address：array(['/datasets/msls/train_val/zurich/query/images/EOC7T_l63Z4LLTSSY6zkkg.jpg,
            # /datasets/msls/train_val/zurich/query/images/AoD5-ZB5YrgyClbR5qmG4g.jpg,
            # /datasets/msls/train_val/zurich/query/images/8VwFgahokEl-0uG-0Yoshg.jpg'], dtype='<U215')
            address_seq = address.split(",")
            key_seq = [key_1.split("/")[-1].split(".")[0] for key_1 in address_seq]
            # need a ',' between keys in the same seq
            # we add it when save to the .csv
            #if len(key_seq) > 1:
            #    key_seq = [key + ',' if i < len(key_seq) - 1 else key for i, key in enumerate(key_seq)]
                    
            keys.append(key_seq)
        keys_all.append(keys)
            
    return keys_all


def write_key_seq(key_col, f):
    """write one key seq for a image/seq to csv"""
    for i, q_key in enumerate(key_col):
        if i > 0:
            f.write(',' + str(q_key))
        else:
            f.write(str(q_key))


def save_to_csv(q_keys, db_keys, path):
    """save the keys in csv, seq_length q_keys match to 20*seq_length db_keys"""
    # create the csv saved keys
    os.makedirs(os.path.join('.', 'files'), exist_ok=True)
    data_file = path
    
    with open(data_file, 'w') as f:
        # one query key match to 5 database keys
        # db_keys size(query_num, 20, seq_length)
        # q_keys size(query_num, 1, seq_length)
        for db_one, q_one in zip(db_keys, q_keys):
            # 20 db col after 1 q_col
            for q_col in q_one:
                # len(q_col) = seq_length
                write_key_seq(q_col, f)
                f.write(' ')
                for db_col in db_one:
                    write_key_seq(db_col, f)
                    f.write(' ')
            f.write("\n")


def get_net(new=False):
    """get the resnet50 and you can change something in it"""
    pretrained_net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    
    # if the pretrained_net is not good, use the net
    if new == False:
        net = pretrained_net
    else:
        net = pretrained_net
        net.fc = nn.Linear(2048, 256, bias=True)
    return net


def load_net(path=None, new=False):
    # load the net used for training
    net = get_net(new)
    
    # load the parameters of the trained net
    net.load_state_dict(torch.load(path))
    net.eval()

    return net

def create_dataset_loader(root_dir, cities, task, seq_length, batch_size):

    # get transform
    meta = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    transform = configure_transform(image_dim = (480, 640), meta = meta)

    # positive are defined within a radius of 25 m 阳性定义在25米的半径范围内
    posDistThr = 25

    # choose subtask to test on [all, s2w, w2s, o2n, n2o, d2n, n2d]
    subtask = 'all'

    val_dataset = MSLS(root_dir, cities = cities, transform = transform, mode = 'test',
                    task = task, seq_length = seq_length, subtask = subtask, posDistThr = posDistThr)
    
    opt = {'batch_size': batch_size}

    # get images
    qLoader = DataLoader(ImagesFromList(val_dataset.qImages[val_dataset.qIdx], transform), **opt)
    dbLoader = DataLoader(ImagesFromList(val_dataset.dbImages, transform), **opt)

    # get positive index (we allow some more slack: default 25 m)
    # pIdx = val_dataset.pIdx

    return val_dataset, qLoader, dbLoader
    

def main(net, device, task, seq_length, out_path, cities, batch_size):
    # the location of msls_dataset in computer
    root_dir = '/datasets/msls'

    # set the size of batch
    # batch_size = batch_size

    # create the datasets and dataloaders   (root_dir, cities, task, seq_length, batch_size)
    val_dataset, qLoader, dbLoader = create_dataset_loader(root_dir, cities, task, seq_length, batch_size)

    print("***load the net successfully")
    # compute the feature of query and database 
    q_feature, q_idx = predict_feature(net, qLoader, device, task.split('2')[0])
    db_feature, _ = predict_feature(net, dbLoader, device, task.split('2')[1])

    print("***compute the features of query and database successfully")

    # compare the q_feature and db_feature and get the index of first 5 similar db_images
    # shape = (len(q_feature), 5)
    db_idx = query_to_dbIdx(q_feature, db_feature).cpu()
    q_idx = q_idx.cpu()

    print("***find the regarding db_idx successfully")

    # find_keys according to the index(index, dataset, mode="query"/"database")
    db_keys = find_keys(db_idx, val_dataset.dbImages)
    q_keys = find_keys(q_idx, val_dataset.qImages)

    print("***find the regarding db_keys successfully")

    # save the keys in rule
    save_to_csv(q_keys, db_keys, out_path)
    print("***save the query keys with regarding db_keys successfully")


if __name__ == "__main__":

    # the location of msls_dataset in computer
    root_dir = Path('/datasets/msls').absolute()
    parser = argparse.ArgumentParser()

    parser.add_argument('--msls-root',
                        type=Path,
                        default=root_dir,
                        help='Path to the dataset')
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
                        help='Choose the gpu to use (cuda:*)')
    parser.add_argument('--model',
                        type=Path,
                        default=None,
                        help='Import the model trained')
    parser.add_argument('--batch-size',
                        type=int,
                        default=150,
                        help='The size of each batch')
    parser.add_argument('--cities',
                        type=str,
                        default="cph,sf",
                        help='The cities to train on')
    parser.add_argument('--output',
                        type=Path,
                        default="./files/my_new_prediction_im2im_val.csv")
    parser.add_argument('--new-net',
                        type=bool,
                        default=False,
                        help='Choose to use the origin resnet50 or not')
    args = parser.parse_args()

    net_path = args.model
    net = load_net(net_path, new=args.new_net)

    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")

    main(net, device, args.task, args.seq_length, args.output, args.cities, args.batch_size)
    