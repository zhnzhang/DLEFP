import torch
import torch.nn as nn
import os
import numpy as np
from sklearn.metrics import f1_score

from transformers.optimization import AdamW
from config import get_train_args
from data import k_fold_split, get_data
from model import GAIN_BERT
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def train(model, trainloader, optimizer, opt):
    model.train()
    # start_time = time.time()

    loss_list = []
    for batch_idx, (ids, labels, words, masks, sentence_id, trigger_id, graphs) in enumerate(trainloader):
        if opt.gpu:
            words = words.cuda()
            masks = masks.cuda()
            sentence_id = sentence_id.cuda()
            trigger_id = trigger_id.cuda()
            graphs = graphs.to('cuda')
            labels = labels.cuda()

        optimizer.zero_grad()

        logit = model(ids=ids,
                      words=words,
                      masks=masks,
                      sentence_id=sentence_id,
                      trigger_id=trigger_id,
                      graphs=graphs)
        loss = nn.functional.cross_entropy(logit, labels)

        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())

    # print("time:%.3f" % (time.time() - start_time))
    return np.mean(loss_list)


def test(model, testloader, opt, filepath=None):
    if filepath is not None:
        f = open(filepath, 'w')
    model.eval()
    total = 0
    correct = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch_idx, (ids, labels, words, masks, sentence_id, trigger_id, graphs) in enumerate(testloader):
            if opt.gpu:
                words = words.cuda()
                masks = masks.cuda()
                sentence_id = sentence_id.cuda()
                trigger_id = trigger_id.cuda()
                graphs = graphs.to('cuda')
                labels = labels.cuda()

            logit = model(ids=ids,
                          words=words,
                          masks=masks,
                          sentence_id=sentence_id,
                          trigger_id=trigger_id,
                          graphs=graphs)
            _, predicted = torch.max(logit.data,1)
            correct += predicted.data.eq(labels.data).cpu().sum()
            y_true += labels.cpu().data.numpy().tolist()
            y_pred += predicted.cpu().data.numpy().tolist()

            if filepath is not None:
                batch = labels.shape[0]
                for i in range(batch):
                    f.write(ids[i] + "\t" + str(labels[i].item()) + "\t" + str(predicted[i].item()) + "\n")

    f1_micro = f1_score(y_true, y_pred, labels=[0, 1, 2], average='micro')
    f1_macro = f1_score(y_true, y_pred, labels=[0, 1, 2], average='macro')
    if filepath is not None:
        f.write("f1_micro: " + str(f1_micro) + "\n")
        f.write("f1_macro: " + str(f1_macro) + "\n")
        f.close()
    return f1_micro, f1_macro


if __name__=='__main__':
    opt = get_train_args()
    if opt.gpu:
        torch.cuda.manual_seed(0)
    else:
        torch.manual_seed(0)

    for i in range(opt.k_fold):
        model_path = opt.model_path + "_" + str(i) + ".pt"
        checkpoint = torch.load(model_path)
        print("best epoch:%d-%d" % (i, checkpoint['epoch']))
