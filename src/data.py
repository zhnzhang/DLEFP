import os
import re
import pickle
from collections import defaultdict
import xml.etree.ElementTree as ET
from sklearn.model_selection import StratifiedKFold

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

import scipy.sparse as sp
import spacy
nlp = spacy.load("en_core_web_trf")


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def k_fold_split(data_path, k_fold):
    # 划分训练集和测试集
    index = []
    labels = []
    label2idx = {}

    tree = ET.parse(data_path)
    root = tree.getroot()
    for document_set in root:
        for document in document_set:
            # label
            label = document.attrib['document_level_value']
            if label not in label2idx:
                label2idx[label] = len(label2idx)
            labels.append(label2idx[label])

    skf = StratifiedKFold(n_splits=k_fold, random_state=0, shuffle=True)
    for train, test in skf.split(np.zeros(len(labels)), labels):
        index.append({'train': train, 'test': test})
    return index, label2idx


class BERTDGLREDataset(Dataset):

    def __init__(self, src_file, save_file, label2idx,
                 index, dataset_type='train', bert_path=None):

        super(BERTDGLREDataset, self).__init__()

        self.data = []
        self.document_data = None
        self.document_max_length = 512

        print('Reading data from {}.'.format(src_file))
        if os.path.exists(save_file):
            with open(file=save_file, mode='rb') as fr:
                info = pickle.load(fr)
                self.document_data = info['data']
            print('load preprocessed data from {}.'.format(save_file))
        else:
            bert = Bert('bert-base-uncased', bert_path)
            self.document_data = []

            # read xml file
            tree = ET.parse(src_file)
            root = tree.getroot()
            for doc in root[0]:
                id = doc.attrib['id']
                label = label2idx[doc.attrib['document_level_value']]

                trigger_list = []
                sentences = []
                Ls = [0]
                L = 0
                for sent in doc:
                    if sent.text == '-EOP- .':
                        continue
                    if len(sent) > 0:
                        tmp = sent.text.lower().split() if sent.text is not None else []
                        trigger_list.append({'sent_id': len(sentences),
                                             'pos': len(tmp),
                                             'global_pos': len(tmp) + Ls[len(sentences)],
                                             'word': sent[0].text.lower(),
                                             'value': label2idx[sent[0].attrib['sentence_level_value']]})

                    s = []
                    for text in sent.itertext():
                        s += text.replace('-EOP- ', '').lower().split()
                    sentences.append(s)
                    L += len(s)
                    Ls.append(L)

                # generate document ids
                words = []
                for sentence in sentences:
                    for word in sentence:
                        words.append(word)

                bert_token, bert_mask, bert_starts, bert_subwords = bert.subword_tokenize_to_ids(words)

                sentence_id = np.zeros((self.document_max_length,), dtype=np.int32)
                trigger_id = np.zeros((self.document_max_length,), dtype=np.int32)
                sentence_num = len(Ls) - 1
                trigger_num = len(trigger_list)

                for idx, v in enumerate(trigger_list, 1):
                    sent_id, pos = v['sent_id'], v['global_pos']

                    pos0 = bert_starts[pos]
                    pos1 = bert_starts[pos + 1]

                    if pos0 >= self.document_max_length - 1:
                        trigger_num = idx - 1
                        break
                    if pos1 >= self.document_max_length - 1:
                        pos1 = self.document_max_length - 1

                    trigger_id[pos0:pos1] = idx

                new_Ls = [1]
                for ii in range(1, len(Ls)):
                    new_Ls.append(bert_starts[Ls[ii]] if Ls[ii] < len(bert_starts) else len(bert_subwords) - 1)
                Ls = new_Ls

                for idx in range(1, len(Ls)):
                    pos0 = Ls[idx - 1]
                    pos1 = Ls[idx]
                    sentence_id[pos0:pos1] = idx
                    if pos1 == 511:
                        sentence_num = idx
                        break
                    if pos0 == 511:
                        sentence_num = idx - 1
                        break

                # construct graph
                graph = self.create_graph(sentence_num, trigger_num, trigger_list)

                assert sentence_num + trigger_num == graph.number_of_nodes() - 1
                assert sentence_num == np.amax(sentence_id)
                assert trigger_num == np.amax(trigger_id)

                self.document_data.append({
                    'ids': id,
                    'labels': label,
                    'triggers': trigger_list,
                    'subwords': bert_subwords,
                    'words': bert_token,
                    'masks': bert_mask,
                    'sentence_id': sentence_id,
                    'trigger_id': trigger_id,
                    'graphs': graph
                })

            # save data
            with open(file=save_file, mode='wb') as fw:
                pickle.dump({'data': self.document_data}, fw)
            print('finish reading {} and save preprocessed data to {}.'.format(src_file, save_file))

        for i in index[dataset_type]:
            self.data.append(self.document_data[i])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]['ids'], \
               torch.tensor(self.data[idx]['labels'], dtype=torch.long), \
               self.data[idx]['words'], self.data[idx]['masks'], \
               torch.tensor(self.data[idx]['sentence_id'], dtype=torch.long), \
               torch.tensor(self.data[idx]['trigger_id'], dtype=torch.long), \
               self.data[idx]['graphs']
        # return self.data[idx]['graphs'], torch.tensor(self.data[idx]['labels'], dtype=torch.long)

    def create_graph(self, sentence_num, trigger_num, trigger_list):

        d = defaultdict(list)

        # add sentence-sentence edges
        for i in range(1, sentence_num + 1):
            d[('node', 'ss', 'node')].append((i, i))  # self-loop
            for j in range(i + 1, sentence_num + 1):
                d[('node', 'ss', 'node')].append((i, j))
                d[('node', 'ss', 'node')].append((j, i))

        # add sentence-trigger edges
        for idx in range(1, trigger_num + 1):
            i = idx + sentence_num
            j = trigger_list[idx - 1]['sent_id'] + 1
            d[('node', 'st', 'node')].append((i, j))
            d[('node', 'st', 'node')].append((j, i))
        if d[('node', 'st', 'node')] == []:
            d[('node', 'st', 'node')].append((0, 1))

        # add global edges
        for i in range(1, sentence_num + trigger_num + 1):
            d[('node', 'global', 'node')].append((0, i))
            d[('node', 'global', 'node')].append((i, 0))
        d[('node', 'global', 'node')].append((0, 0))

        graph = dgl.heterograph(d)
        # print(graph)

        assert len(graph.etypes) == 3, "etypes wrong"

        return graph


class MyDataset(Dataset):

    def __init__(self, src_file, save_file, label2idx,
                 dataset_index, dataset_type='train', bert_path=None):

        super(MyDataset, self).__init__()

        self.data = []
        self.document_data = None
        self.document_max_length = 512
        self.sentence_max_length = 150
        self.max_word_num = 0

        print('Reading data from {}.'.format(src_file))
        if os.path.exists(save_file):
            with open(file=save_file, mode='rb') as fr:
                info = pickle.load(fr)
                self.document_data = info['data']
            print('load preprocessed data from {}.'.format(save_file))
        else:
            tok = BertTokenizer.from_pretrained(bert_path)
            self.document_data = []

            # read xml file
            tree = ET.parse(src_file)
            root = tree.getroot()
            for doc in root[0]:
                id = doc.attrib['id']
                label = label2idx[doc.attrib['document_level_value']]

                # for doc
                sentence_list = []
                trigger_list = []
                for sent in doc:
                    if sent.text == '-EOP- .':
                        continue

                    trig_in1sent_info = []
                    s = sent.text.replace("``", '"').replace("''", '"').replace('__', '_') if sent.text is not None else ''
                    for event in sent:
                        trigger = event.text.replace("``", '"').replace("''", '"').replace('__', '_')
                        assert len(nlp(trigger)) == 1
                        trig_in1sent_info.append({'index': len(nlp(s)),
                                                  'token': trigger.lower(),
                                                  'value': label2idx[event.attrib['sentence_level_value']]})

                        tail = event.tail.replace("``", '"').replace("''", '"').replace('__', '_') if event.tail is not None else ''
                        s += trigger + tail
                    s = s.replace('-EOP- ', '')
                    if len(s.split()) <= 3:
                        continue

                    # for sent
                    sent_subwords = []
                    word_list = {'token': [], 'idx': []}
                    edges_src = []
                    edges_dst = []

                    parsed_data = nlp(s)
                    for token in parsed_data:
                        word = token.text.lower()
                        word_subwords = tok.tokenize(word)
                        pos0 = 1 + len(sent_subwords)
                        pos1 = pos0 + len(word_subwords)
                        if pos0 >= self.sentence_max_length - 1:
                            break
                        if pos1 > self.sentence_max_length - 1:
                            pos1 = self.sentence_max_length - 1

                        word_idx = torch.zeros(self.sentence_max_length)
                        word_idx[pos0:pos1] = 1.0 / (pos1 - pos0)
                        word_list['token'].append(word)
                        word_list['idx'].append(word_idx)

                        if token.head.i != token.i:
                            edges_src.append(token.head.i)
                            edges_dst.append(token.i)

                        sent_subwords += word_subwords

                    for t in trig_in1sent_info:
                        index = t['index']
                        if index >= len(word_list['token']):
                            break
                        assert t['token'] == word_list['token'][index]
                        trigger_list.append({'sent_id': len(sentence_list),
                                             'index': index,
                                             'idx': word_list['idx'][index],
                                             'value': t['value']})

                    data = tok(s.lower(), return_tensors='pt', padding='max_length', truncation=True, max_length=150)
                    sent_idx = torch.stack(word_list['idx'], dim=0)
                    sent_dep_adj = self.create_dep_adj(edges_src, edges_dst, len(word_list['idx']))
                    sentence_list.append({'ids': data['input_ids'],
                                          'mask': data['attention_mask'],
                                          'idx': sent_idx,
                                          'dep_adj': sent_dep_adj})
                    if len(sentence_list) >= 35:
                        break

                d = ''
                for sent in doc:
                    if len(sent) > 0:
                        s = ''
                        for t in sent.itertext():
                            s += t
                        s = s.replace('-EOP- ', '').replace("``", '"').replace("''", '"').replace('__', '_').lower()
                        d += s + ' '
                doc_data = tok(d, return_tensors='pt', padding='max_length',
                                   truncation=True, max_length=512)

                # construct graph
                graph = self.create_graph(sentence_list, trigger_list)
                assert graph.number_of_nodes() == len(sentence_list) + len(trigger_list) + 1

                self.document_data.append({
                    'id': id,
                    'label': label,
                    'doc_ids': doc_data['input_ids'],
                    'doc_mask': doc_data['attention_mask'],
                    'max_word_num': self.max_word_num,
                    'sentences': sentence_list,
                    'triggers': trigger_list,
                    'graph': graph
                })
                self.max_word_num = 0

            # save data
            with open(file=save_file, mode='wb') as fw:
                pickle.dump({'data': self.document_data}, fw)
            print('finish reading {} and save preprocessed data to {}.'.format(src_file, save_file))

        for i in dataset_index[dataset_type]:
            self.data.append(self.document_data[i])
            if self.max_word_num < self.document_data[i]['max_word_num']:
                self.max_word_num = self.document_data[i]['max_word_num']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        max_word_num = self.max_word_num
        sentence_list = self.data[idx]['sentences']
        sent_ids = []
        sent_mask = []
        sent_idx = []
        sent_dep_adj = []
        for s in sentence_list:
            sent_ids.append(s['ids'])
            sent_mask.append(s['mask'])

            s_idx = s['idx']
            word_num = s_idx.shape[0]
            s_idx = torch.cat((s_idx, torch.zeros([max_word_num - word_num, self.sentence_max_length])), dim=0)
            sent_idx.append(s_idx)

            dep_adj = s['dep_adj']
            assert dep_adj.shape[0] == word_num
            dep_adj = F.pad(dep_adj, [0, max_word_num - word_num, 0, max_word_num - word_num])
            sent_dep_adj.append(dep_adj)

        sent_ids = torch.cat(sent_ids, dim=0)  # [seq_num, seq_len]
        sent_mask = torch.cat(sent_mask, dim=0)  # [seq_num, seq_len]
        sent_idx = torch.stack(sent_idx, dim=0)  # [seq_num, word_num, seq_len]
        sent_dep_adj = torch.stack(sent_dep_adj, dim=0)  # [seq_num, word_num, word_num]

        trigger_list = self.data[idx]['triggers']
        trigger_sid = []
        trigger_index = []
        trigger_label = []
        for t in trigger_list:
            trigger_sid.append(t['sent_id'])
            trigger_index.append(t['index'])
            assert (sent_idx[t['sent_id']][t['index']] != t['idx']).sum() == 0
            trigger_label.append(t['value'])
        trigger_sid = torch.tensor(trigger_sid)  # [trigger_num]
        trigger_index = torch.tensor(trigger_index)  # [trigger_num]
        trigger_label = torch.tensor(trigger_label)  # [trigger_num]

        return self.data[idx]['id'], \
               torch.tensor(self.data[idx]['label'], dtype=torch.long), \
               self.data[idx]['doc_ids'], \
               self.data[idx]['doc_mask'], \
               sent_ids, \
               sent_mask, \
               sent_idx.to('cuda:1'), \
               sent_dep_adj.to('cuda:1'), \
               trigger_sid.to('cuda:1'), \
               trigger_index.to('cuda:1'), \
               trigger_label, \
               self.data[idx]['graph']

    def create_dep_adj(self, edges_src, edges_dst, num_nodes):
        if self.max_word_num < num_nodes:
            self.max_word_num = num_nodes

        adj = sp.coo_matrix((np.ones(len(edges_src)), (edges_src, edges_dst)),
                            shape=(num_nodes, num_nodes),
                            dtype=np.float32)
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = normalize(adj + sp.eye(adj.shape[0]))
        adj = sparse_mx_to_torch_sparse_tensor(adj).to_dense()
        return adj

    def create_graph(self, sentence_list, trigger_word_list):
        d = defaultdict(list)
        sent_num = len(sentence_list)
        trigger_num = len(trigger_word_list)

        # add neighbor edges
        for i in range(1, sent_num):
            d[('node', 'neighbor', 'node')].append((i, i + 1))
            d[('node', 'neighbor', 'node')].append((i + 1, i))

        # add global edges
        for i in range(1, sent_num + 1):
            d[('node', 'global', 'node')].append((0, i))
            d[('node', 'global', 'node')].append((i, 0))

        # add trigger-doc, trigger-sent edges
        '''
        for s in sentence_list:
            if s['trigger'] == False:
                continue
            d[('node', 'trigger', 'node')].append((s['sent_id'] + 1, 0))  # uni-direction'''
        for i in range(trigger_num):
            j = i + sent_num + 1
            # trigger-doc
            d[('node', 'td', 'node')].append((0, j))
            d[('node', 'td', 'node')].append((j, 0))

            # trigger_sent
            d[('node', 'ts', 'node')].append((j, trigger_word_list[i]['sent_id'] + 1))
            d[('node', 'ts', 'node')].append((trigger_word_list[i]['sent_id'] + 1, j))

        graph = dgl.heterograph(d)
        # print(graph)
        assert len(graph.etypes) == 4, "etypes wrong"

        return graph


class ChineseDataset(Dataset):

    def __init__(self, src_file, save_file, label2idx,
                 index, dataset_type='train', bert_path=None):

        super(ChineseDataset, self).__init__()

        self.data = []
        self.document_data = None
        self.document_max_length = 512
        self.sentence_max_length = 150

        print('Reading data from {}.'.format(src_file))
        if os.path.exists(save_file):
            with open(file=save_file, mode='rb') as fr:
                info = pickle.load(fr)
                self.document_data = info['data']
            print('load preprocessed data from {}.'.format(save_file))
        else:
            bert = Bert(bert_path)
            tok = BertTokenizer.from_pretrained(bert_path)
            self.document_data = []

            # read xml file
            tree = ET.parse(src_file)
            root = tree.getroot()
            for doc in root[0]:
                id = doc.attrib['id']
                label = label2idx[doc.attrib['document_level_value']]

                sentence_list = []
                trigger_word_list = []
                flag = False
                for sent in doc:
                    if sent.text == '-EOP-.' or sent.text == '。':
                        continue  # 去掉无意义的句子，包括"-EOP."和"。"

                    s = ''  # 一个句子
                    for t in sent.itertext():
                        s += t
                    s = s.replace('-EOP-.', '。').lower()

                    if re.match(r'\d{4}\D\d{2}\D\d{2}\D\d{2}:\d{2}\D$', s) is not None:
                        flag = True
                        continue  # 去掉日期
                    elif flag:
                        flag = False
                        if len(sent) == 0:
                            continue  # 去掉不含主题事件的日期的下一行
                        # print(s)
                    if len(s) <= 4:
                        continue  # 去掉少于4个字的句子

                    data = tok(s, return_tensors='pt', padding='max_length', truncation=True, max_length=150)
                    # 中文BERT按字分词
                    # s_token, s_mask, s_starts, s_subwords = bert.subword_tokenize_to_ids(s.split())
                    # assert 0 == (data['input_ids'] != s_token).sum()
                    sent_info = {'trigger_num': 0,
                                 'data': data['input_ids'],
                                 'attention': data['attention_mask']}
                    if len(sent) > 0:
                        # has triggers
                        tmp = sent.text.lower() if sent.text is not None else ''
                        for event in sent:
                            tmp_subwords = tok.tokenize(tmp)
                            trigger_subwords = tok.tokenize(event.text.lower())
                            pos0 = len(tmp_subwords) + 1
                            pos1 = pos0 + len(trigger_subwords)
                            if pos0 >= self.sentence_max_length - 1:
                                break
                            if pos1 >= self.sentence_max_length - 1:
                                pos1 = self.sentence_max_length - 1
                            else:
                                assert tok.convert_ids_to_tokens(data['input_ids'][0, pos0:pos1]) == trigger_subwords

                            trigger_word_idx = torch.zeros(self.sentence_max_length)
                            trigger_word_idx[pos0:pos1] = 1.0 / (pos1 - pos0)

                            trigger_word_list.append({'sent_id': len(sentence_list),
                                                      'idx': trigger_word_idx,
                                                      'value': label2idx[event.attrib['sentence_level_value']]})
                            tmp += event.text.lower()
                            if event.tail is not None:
                                tmp += event.tail.lower()

                            sent_info['trigger_num'] += 1
                    sentence_list.append(sent_info)
                    if len(sentence_list) >= 35:
                        break

                trigger = ''
                for sent in doc:
                    if len(sent) > 0:
                        s = ''
                        for t in sent.itertext():
                            s += t
                        s = s.replace('-EOP-.', '。').lower()
                        trigger += s
                trigger_data = tok(trigger, return_tensors='pt', padding='max_length',
                                   truncation=True, max_length=512)

                # construct graph
                graph = self.create_graph(sentence_list, trigger_word_list)

                assert graph.number_of_nodes() == len(sentence_list) + len(trigger_word_list) + 1

                self.document_data.append({
                    'ids': id,
                    'labels': label,
                    'triggers': trigger_data['input_ids'],
                    'trigger_masks': trigger_data['attention_mask'],
                    'sentences': sentence_list,
                    'trigger_words': trigger_word_list,
                    'graphs': graph
                })

            # save data
            with open(file=save_file, mode='wb') as fw:
                pickle.dump({'data': self.document_data}, fw)
            print('finish reading {} and save preprocessed data to {}.'.format(src_file, save_file))

        for i in index[dataset_type]:
            self.data.append(self.document_data[i])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence_list = self.data[idx]['sentences']
        data_list = []
        attention_list = []
        for s in sentence_list:
            data_list.append(s['data'])
            attention_list.append(s['attention'])
        data = torch.cat(data_list, dim=0)
        attention = torch.cat(attention_list, dim=0)

        trigger_word_list = self.data[idx]['trigger_words']
        sent_idx_list = []
        trigger_word_idx_list = []
        trigger_label_list = []
        for t in trigger_word_list:
            sent_idx_list.append(t['sent_id'])
            trigger_word_idx_list.append(t['idx'])
            trigger_label_list.append(t['value'])
        sent_idx = torch.tensor(sent_idx_list).cuda()
        trigger_word_idx = torch.stack(trigger_word_idx_list, dim=0).cuda()
        trigger_label = torch.tensor(trigger_label_list)

        return self.data[idx]['ids'], \
               torch.tensor(self.data[idx]['labels'], dtype=torch.long), \
               self.data[idx]['triggers'], \
               self.data[idx]['trigger_masks'], \
               data, \
               attention, \
               sent_idx, \
               trigger_word_idx, \
               trigger_label, \
               self.data[idx]['graphs']
        # return self.data[idx]['graphs'], torch.tensor(self.data[idx]['labels'], dtype=torch.long)

    def create_graph(self, sentence_list, trigger_word_list):

        d = defaultdict(list)
        sent_num = len(sentence_list)
        trigger_num = len(trigger_word_list)

        # add neighbor edges
        for i in range(1, sent_num):
            d[('node', 'neighbor', 'node')].append((i, i + 1))
            d[('node', 'neighbor', 'node')].append((i + 1, i))

        # add global edges
        for i in range(1, sent_num + 1):
            d[('node', 'global', 'node')].append((0, i))
            d[('node', 'global', 'node')].append((i, 0))

        # add trigger-doc, trigger-sent edges
        '''
        for s in sentence_list:
            if s['trigger'] == False:
                continue
            d[('node', 'trigger', 'node')].append((s['sent_id'] + 1, 0))  # uni-direction'''
        for i in range(trigger_num):
            j = i + sent_num + 1
            # trigger-doc
            d[('node', 'td', 'node')].append((0, j))
            d[('node', 'td', 'node')].append((j, 0))

            # trigger_sent
            d[('node', 'ts', 'node')].append((j, trigger_word_list[i]['sent_id'] + 1))
            d[('node', 'ts', 'node')].append((trigger_word_list[i]['sent_id'] + 1, j))

        graph = dgl.heterograph(d)
        # print(graph)

        assert len(graph.etypes) == 4, "etypes wrong"

        return graph


class Bert():
    MASK = '[MASK]'
    CLS = "[CLS]"
    SEP = "[SEP]"

    def __init__(self, model_path=None):
        super().__init__()
        # self.model_name = model_name
        # print(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.max_len = 150

    def tokenize(self, text, masked_idxs=None):
        tokenized_text = self.tokenizer.tokenize(text)
        if masked_idxs is not None:
            for idx in masked_idxs:
                tokenized_text[idx] = self.MASK
        # prepend [CLS] and append [SEP]
        # see https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_classifier.py#L195  # NOQA
        tokenized = [self.CLS] + tokenized_text + [self.SEP]
        return tokenized

    def tokenize_to_ids(self, text, masked_idxs=None, pad=True):
        tokens = self.tokenize(text, masked_idxs)
        return tokens, self.convert_tokens_to_ids(tokens, pad=pad)

    def convert_tokens_to_ids(self, tokens, pad=True):
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        ids = torch.tensor([token_ids])
        # assert ids.size(1) < self.max_len
        ids = ids[:, :self.max_len]  # https://github.com/DreamInvoker/GAIN/issues/4
        if pad:
            padded_ids = torch.zeros(1, self.max_len).to(ids)
            padded_ids[0, :ids.size(1)] = ids
            mask = torch.zeros(1, self.max_len).to(ids)
            mask[0, :ids.size(1)] = 1
            return padded_ids, mask
        else:
            return ids

    def flatten(self, list_of_lists):
        for list in list_of_lists:
            for item in list:
                yield item

    def subword_tokenize(self, tokens):
        """Segment each token into subwords while keeping track of
        token boundaries.
        Parameters
        ----------
        tokens: A sequence of strings, representing input tokens.
        Returns
        -------
        A tuple consisting of:
            - A list of subwords, flanked by the special symbols required
                by Bert (CLS and SEP).
            - An array of indices into the list of subwords, indicating
                that the corresponding subword is the start of a new
                token. For example, [1, 3, 4, 7] means that the subwords
                1, 3, 4, 7 are token starts, while all other subwords
                (0, 2, 5, 6, 8...) are in or at the end of tokens.
                This list allows selecting Bert hidden states that
                represent tokens, which is necessary in sequence
                labeling.
        """
        subwords = list(map(self.tokenizer.tokenize, tokens))
        subword_lengths = list(map(len, subwords))
        subwords = [self.CLS] + list(self.flatten(subwords))[:148] + [self.SEP]
        token_start_idxs = 1 + np.cumsum([0] + subword_lengths[:-1])
        token_start_idxs[token_start_idxs > 148] = 149
        return subwords, token_start_idxs

    def subword_tokenize_to_ids(self, tokens):
        """Segment each token into subwords while keeping track of
        token boundaries and convert subwords into IDs.
        Parameters
        ----------
        tokens: A sequence of strings, representing input tokens.
        Returns
        -------
        A tuple consisting of:
            - A list of subword IDs, including IDs of the special
                symbols (CLS and SEP) required by Bert.
            - A mask indicating padding tokens.
            - An array of indices into the list of subwords. See
                doc of subword_tokenize.
        """
        subwords, token_start_idxs = self.subword_tokenize(tokens)
        subword_ids, mask = self.convert_tokens_to_ids(subwords)
        return subword_ids, mask, token_start_idxs, subwords

    def segment_ids(self, segment1_len, segment2_len):
        ids = [0] * segment1_len + [1] * segment2_len
        return torch.tensor([ids])


def collate(samples):
    # only consider batch_size=1
    id, label, doc_ids, doc_mask, \
    sent_ids, sent_mask, sent_idx, sent_dep_adj, \
    trigger_sid, trigger_index, trigger_label, graph = map(list, zip(*samples))

    batched_id = tuple(id)
    batched_label = torch.tensor(label)  # [bsz]
    batched_doc_ids = torch.cat(doc_ids, dim=0)  # [bsz, doc_len]
    batched_doc_mask = torch.cat(doc_mask, dim=0)  # [bsz, doc_len]

    batched_sent_ids = torch.cat(sent_ids, dim=0)  # [bsz * seq_num, seq_len]
    bacthed_sent_mask = torch.cat(sent_mask, dim=0)  # [bsz * seq_num, seq_len]
    batched_sent_idx = torch.cat(sent_idx, dim=0)  # bsz * [seq_num, word_num, seq_len]
    batched_sent_dep_adj = torch.cat(sent_dep_adj, dim=0)  # bsz * [seq_num, word_num, word_num]

    batched_trigger_sid = trigger_sid  # bsz * [trigger_num]
    batched_trigger_index = trigger_index  # bsz * [trigger_num]
    batched_trigger_label = torch.cat(trigger_label, dim=0)  # [bsz * trigger_num]
    batched_graph = dgl.batch(graph)
    return batched_id, batched_label, batched_doc_ids, batched_doc_mask, \
           batched_sent_ids, bacthed_sent_mask, batched_sent_idx, batched_sent_dep_adj, \
           batched_trigger_sid, batched_trigger_index, batched_trigger_label, \
           batched_graph


def get_data(opt, label2idx, index):
    trainset = MyDataset(opt.data_path, opt.data_save_path, label2idx,
                         index, dataset_type='train', bert_path=opt.bert_path)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size,
                             shuffle=True,
                             collate_fn=collate)
    testset = MyDataset(opt.data_path, opt.data_save_path, label2idx,
                        index, dataset_type='test', bert_path=opt.bert_path)
    testloader = DataLoader(testset, batch_size=opt.test_batch_size,
                            shuffle=False,
                            collate_fn=collate)
    return trainloader, testloader


if __name__ == '__main__':
    index, label2idx = k_fold_split("../data/dlef_corpus/train.xml", 10)
    train_set = MyDataset('../data/dlef_corpus/train.xml', '../data/train.pkl', label2idx, index[0],
                          dataset_type='train', bert_path="../../data/bert-base-uncased")
    _ = train_set.__getitem__(0)
    dataloader = DataLoader(train_set, batch_size=2, shuffle=False, collate_fn=collate)
    for _ in dataloader:
        # g_unbatch = dgl.unbatch(j1)
        # n = g_unbatch[0].number_of_nodes('node')
        print("hello")
    print("end")
