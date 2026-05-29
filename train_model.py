import torch   进口火炬
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
from tqdm import tqdm
import ast
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
from sklearn.metrics import average_precision_score
from dataload import GET_DATA_PPL
from model import Model

torch.cuda.empty_cache()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train(epoch):
    print("------------------Training Epoch: ", epoch)
    model.train()
    return_predictions = []
    return_labels = []
    for batch_idx, sample in enumerate(tqdm(train_loader)):
        article_id, label, input_sentences = sample['article_id'],sample['label_np'],sample['input_sentences']
        diff_3 = sample['diff_3']
        sentence_feature = model.extract_deberta_PPL(text=input_sentences,diff_3=diff_3,batchsize=args.batch_size)
        label = [list(row) for row in zip(*label)]
        label = torch.stack([torch.stack(sublist) for sublist in label])
        label = label.type(LongTensor)
        optimizer.zero_grad()
        output = model(sentence_feature)
       
        loss = F.binary_cross_entropy_with_logits(output, label.float())
        loss.backward()
        optimizer.step()

        label_np = label.cpu().numpy()
        prediction_np = output.data.cpu().numpy()

        return_predictions.append(prediction_np)
        return_labels.append(label_np)
    predictions_np, labels_np = np.concatenate(return_predictions), np.concatenate(return_labels)
    mAP = average_precision_score(labels_np, sigmoid(predictions_np))
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAP: {:.6f} \n'.format(
        epoch, batch_idx * len(label), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), mAP))
    f_logger.write("epoch-{}: train_loss: {:.4f}; train_mAP: {:.4f} \n".format(epoch, loss, mAP))

def val():
    global val_loader
    return_predictions = []
    return_labels = []
    with torch.no_grad():
        model.eval()
        val_loss = 0
        for batch_idx, sample in enumerate(tqdm(val_loader)):
            article_id, label, input_sentences = sample['article_id'],sample['label_np'],sample['input_sentences']
            diff_3 = sample['diff_3']
            sentence_feature = model.extract_deberta_PPL(text=input_sentences,diff_3=diff_3,batchsize=1)
            label = [list(row) for row in zip(*label)]
            label = torch.stack([torch.stack(sublist) for sublist in label])
            label = label.type(LongTensor)
            optimizer.zero_grad()
            output = model(sentence_feature) 

            val_loss += BCE_criterion(output, label.float())
            predicted = output.data
            predicted_np = predicted.cpu().numpy()
            label_np = label.cpu().numpy()

            return_predictions.append(predicted_np)
            return_labels.append(label_np)

        predictions_np, labels_np = np.concatenate(return_predictions), np.concatenate(return_labels)

        mAP = average_precision_score(labels_np, sigmoid(predictions_np))
        val_loss /= len(val_loader)

        print('\nValidation set: Average loss: {:.4f}, mAP: {:.4f} \n'
              .format(val_loss, mAP))
    return labels_np, output.data, val_loss, mAP

def main(args):
    global BCE_criterion, train_loader, val_loader, model, optimizer,f_logger
    BCE_criterion = nn.BCEWithLogitsLoss()

    log_name = os.path.join("windows_log", args.model_name)
    os.makedirs(log_name, exist_ok=True)

    traindata = GET_DATA_PPL(file_path=args.train_file)
    valdata = GET_DATA_PPL(file_path=args.val_file)
    
    train_loader = torch.utils.data.DataLoader(traindata, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu)
    val_loader = torch.utils.data.DataLoader(valdata, batch_size=1, shuffle=False, num_workers=args.n_cpu)

    model = Model(num_labels=args.sentences_in_window,
                                deberta_detector_name=args.deberta_detector_name,
                                ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=0, verbose=True)

    best_macro_mAP = 0

    f_logger = open(log_name + "/logger_info.txt", 'w')
    for epoch_org in range(args.num_epoch):
        epoch = epoch_org + 1
        train(epoch)
        _, _, val_loss, macro_mAP = val()
        scheduler.step(macro_mAP)

        f_logger.write("epoch-{}: val: {:.4f}; mAP: {:.4f} \n".format(epoch, val_loss, macro_mAP))
        if macro_mAP > best_macro_mAP:
            best_macro_mAP = macro_mAP
            torch.save(model, log_name + "/epoch-val-best.pkl")
            best_epoch = epoch
        if epoch == args.num_epoch:
            torch.save(model, log_name + "/epoch-last.pkl")

    f_logger.write("best epoch num: %d" % best_epoch)
    f_logger.close()

    results = vars(args)
    results.update({'best_epoch_mAP': best_macro_mAP, 'best_epoch': best_epoch})

    with open(os.path.join(log_name, "train_info.json"), 'w') as f:
        json.dump(results, f, indent=2)


if __name__=="__main__":
    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='model name')
    parser.add_argument('--train_file', type=str)
    parser.add_argument('--val_file', type=str)
    parser.add_argument('--n_train_sample', type=int, default=10000, help="number of training samples")
    parser.add_argument('--n_val_sample', type=int, default=1000, help="number of test samples")
    parser.add_argument('--sentences_in_window', type=int, help="number of sentences with in the receptive field")
    parser.add_argument('--deberta_detector_name', type=str, default="D:\\wy\\deberta-v3-large\\", help="sentence feature encoder")
    parser.add_argument('--num_epoch', type=int, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, help='size of the batches') 
    parser.add_argument('--lr', type=float, help='learning rate')


    parser.add_argument('--n_cpu', type=int,help='number of cpu threads to use during batch generation')
    parser.add_argument('--save_interval', type=int, default=1, help='the interval between saved epochs')
    parser.add_argument('--process_interval', type=int, default=2, help='the interval between process print')

    args = parser.parse_args()
    print(args)

    main(args)










