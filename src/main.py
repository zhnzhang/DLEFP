import torch
import torch.nn as nn
import os
import numpy as np
from sklearn.metrics import f1_score

from transformers.optimization import AdamW
from config import get_train_args
from data import k_fold_split, get_data
from model import GAIN_BERT
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'


def train(model, trainloader, optimizer, opt):
    model.train()
    # start_time = time.time()

    loss_list = []
    for batch_idx, (ids, labels, triggers, trigger_masks, words, masks,
                    sent_idx, trigger_word_idx, trigger_labels, graphs) in \
            enumerate(trainloader):
        if opt.gpu:
            triggers = triggers.cuda()
            trigger_masks = trigger_masks.cuda()
            words = words.cuda()
            masks = masks.cuda()
            # sent_idx = sent_idx.cuda()
            # trigger_word_idx = trigger_word_idx.cuda()
            graphs = graphs.to('cuda')
            labels = labels.cuda()
            trigger_labels = trigger_labels.cuda()

        optimizer.zero_grad()

        logit, trigger_logit = model(ids=ids,
                                    triggers=triggers,
                                    trigger_masks=trigger_masks,
                                    words=words,
                                    masks=masks,
                                    sent_idx=sent_idx,
                                    trigger_word_idx=trigger_word_idx,
                                    graphs=graphs)
        main_loss = nn.functional.cross_entropy(logit, labels)
        aux_loss = nn.functional.cross_entropy(trigger_logit, trigger_labels)
        loss = main_loss + opt.labmda * aux_loss

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
        for batch_idx, (ids, labels, triggers, trigger_masks, words, masks,
                        sent_idx, trigger_word_idx, trigger_labels, graphs) in \
                enumerate(testloader):
            if opt.gpu:
                triggers = triggers.cuda()
                trigger_masks = trigger_masks.cuda()
                words = words.cuda()
                masks = masks.cuda()
                # sent_idx = sent_idx.cuda()
                # trigger_word_idx = trigger_word_idx.cuda()
                graphs = graphs.to('cuda')
                labels = labels.cuda()
                trigger_labels = trigger_labels.cuda()

            logit, _ = model(ids=ids,
                             triggers=triggers,
                             trigger_masks=trigger_masks,
                             words=words,
                             masks=masks,
                             sent_idx=sent_idx,
                             trigger_word_idx=trigger_word_idx,
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

    index, label2idx = k_fold_split(opt.data_path, opt.k_fold)
    f1_micro_list = []
    f1_macro_list = []
    for i in range(opt.k_fold):
        model_path = opt.model_path + "_" + str(i) + ".pt"
        output_path = opt.output_path + "_" + str(i) + ".txt"
        trainloader, testloader = get_data(opt, label2idx, index[i])
        model = GAIN_BERT(opt, len(label2idx))
        if opt.gpu:
            # model = nn.DataParallel(model)
            model.cuda()
        optimizer = AdamW(model.parameters(), lr=opt.lr)

        max_f1 = 0
        for epoch in range(opt.nepoch):
            train_loss = train(model, trainloader, optimizer, opt)
            test_f1_micro, test_f1_macro = test(model, testloader, opt)
            print("Epoch:%d-%d loss:%f F1_micro:%.2f F1_macro:%.2f" % (i, epoch, train_loss,
                                                                       test_f1_micro * 100, test_f1_macro * 100))
            if test_f1_micro + test_f1_macro > max_f1:
                max_f1 = test_f1_micro + test_f1_macro
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, model_path)

        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        test_f1_micro, test_f1_macro = test(model, testloader, opt, filepath=output_path)
        print("Epoch:%d-%d F1_micro:%.2f F1_macro:%.2f" % (i, checkpoint['epoch'],
                                                           test_f1_micro * 100, test_f1_macro * 100))
        f1_micro_list.append(test_f1_micro)
        f1_macro_list.append(test_f1_macro)

    output = open(opt.output_path + ".txt", "w")
    f1_micro_a = np.mean(f1_micro_list)
    f1_macro_a = np.mean(f1_macro_list)
    output.write("batch_size=" + str(opt.batch_size) + "\n")
    output.write("lr=" + str(opt.lr) + "\n")
    output.write("f1_micro_a: " + str(f1_micro_a) + "\n")
    output.write("f1_macro_a: " + str(f1_macro_a) + "\n")
    print("F1_micro_a: %.2f F1_macro_a: %.2f" % (f1_micro_a * 100, f1_macro_a * 100))

    ct_p = []
    ct_m = []
    ps_p = []
    for i in range(opt.k_fold):
        filename = opt.output_path + "_" + str(i) + ".txt"
        y_true = []
        y_pred = []
        with open(filename, "r") as f:
            for l in f.readlines():
                line = l.split()
                if len(line) < 3:
                    break
                y_true.append(line[1])
                y_pred.append(line[2])

        with open(filename, "a") as f:
            t_ct_p = f1_score(y_true, y_pred, labels=[0], average="macro")
            f.write("CT+: " + str(t_ct_p) + "\n")
            ct_p.append(t_ct_p)

            t_ct_m = f1_score(y_true, y_pred, labels=[1], average="macro")
            f.write("CT-: " + str(t_ct_m) + "\n")
            ct_m.append(t_ct_m)

            t_ps_p = f1_score(y_true, y_pred, labels=[2], average="macro")
            f.write("PS+: " + str(t_ps_p) + "\n")
            ps_p.append(t_ps_p)

    output.write("CT+: " + str(np.mean(ct_p)) + "\n")
    output.write("CT-: " + str(np.mean(ct_m)) + "\n")
    output.write("PS+: " + str(np.mean(ps_p)) + "\n")
    output.close()
