import os
import torch
from pathlib import Path
import predict
import evaluate
    

def save_evaluate(args, net, epoch, i=999999, cities='cph,sf'):
    outpath = args.out_path

    # predict and save the keys
    print(f'\nStart to predict the keys of cities: {cities}')
    predict_path = Path(os.path.join(outpath, Path(f"prediction_{cities}_val_epoch{epoch + 1}_i{i}.csv")))
    predict.main(args, net, predict_path, cities)
    print(f'save the prediction successfully!!')

    # evaluate the predictions above and save the results
    print(f'\nStart to evaluate the prediction of cities: {cities}')
    evaluate_path = Path(os.path.join(outpath, Path(f"evaluate_task{cities}_epoch{epoch + 1}_i{i}.csv")))
    recall_1 = evaluate.main(args, predict_path, evaluate_path, cities)
    print(f'evaluate the model sucessfully! you can see the result in {outpath}\n')
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