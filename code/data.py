import json
import math
import os
import pickle
import random
from collections import defaultdict
import xml.etree.ElementTree as ET

import dgl
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
from transformers import BertTokenizer, BertModel


def k_fold_split(data_path):
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

    skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    for train, test in skf.split(np.zeros(len(labels)), labels):
        index.append({'train': train, 'test': test})
    return index, label2idx


class BERTDGLREDataset(Dataset):

    def __init__(self, src_file, save_file, word2id, ner2id, rel2id,
                 dataset_type='train', instance_in_train=None, opt=None):

        super(BERTDGLREDataset, self).__init__()

        # record training set mention triples
        self.instance_in_train = set([]) if instance_in_train is None else instance_in_train
        self.data = None
        self.document_max_length = 512
        self.INFRA_EDGE = 0
        self.INTER_EDGE = 1
        self.LOOP_EDGE = 2
        self.count = 0

        print('Reading data from {}.'.format(src_file))
        if os.path.exists(save_file):
            with open(file=save_file, mode='rb') as fr:
                info = pickle.load(fr)
                self.data = info['data']
                self.instance_in_train = info['intrain_set']
            print('load preprocessed data from {}.'.format(save_file))

        else:
            bert = Bert(BertModel, 'bert-base-uncased', opt.bert_path)

            # read xml file
            tree = ET.parse(src_file)
            root = tree.getroot()
            for i in index:
                doc = root[0][i]
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
                        tmp = sent.text.lower().split()
                        trigger_list.append({'pos': len(tmp),
                                             'sent_id': len(sentences),
                                             'global_pos': len(tmp) + Ls[len(sentences)],
                                             'word': sent[0].text.lower(),
                                             'value': label2idx[sent[0].attrib['sentence_level_value']]})

                    s = []
                    for text in sent.itertext():
                        s += text.replace('-EOP- ', '').lower().split()
                    sentences.append(s)
                    L += len(s)
                    Ls.append(L)

                # generate positive examples
                train_triple = []
                new_labels = []
                for label in labels:
                    head, tail, relation, evidence = label['h'], label['t'], label['r'], label['evidence']
                    assert (relation in rel2id), 'no such relation {} in rel2id'.format(relation)
                    label['r'] = rel2id[relation]

                    train_triple.append((head, tail))

                    label['in_train'] = False

                    # record training set mention triples and mark it for dev and test set
                    for n1 in entity_list[head]:
                        for n2 in entity_list[tail]:
                            mention_triple = (n1['name'], n2['name'], relation)
                            if dataset_type == 'train':
                                self.instance_in_train.add(mention_triple)
                            else:
                                if mention_triple in self.instance_in_train:
                                    label['in_train'] = True
                                    break

                    new_labels.append(label)

                # generate negative examples
                na_triple = []
                for j in range(len(entity_list)):
                    for k in range(len(entity_list)):
                        if j != k and (j, k) not in train_triple:
                            na_triple.append((j, k))

                # generate document ids
                words = []
                for sentence in sentences:
                    for word in sentence:
                        words.append(word)

                bert_token, bert_starts, bert_subwords = bert.subword_tokenize_to_ids(words)

                word_id = np.zeros((self.document_max_length,), dtype=np.int32)
                pos_id = np.zeros((self.document_max_length,), dtype=np.int32)
                ner_id = np.zeros((self.document_max_length,), dtype=np.int32)
                mention_id = np.zeros((self.document_max_length,), dtype=np.int32)
                word_id[:] = bert_token[0]

                entity2mention = defaultdict(list)
                mention_idx = 1
                already_exist = set()
                for idx, vertex in enumerate(entity_list, 1):
                    for v in vertex:

                        sent_id, (pos0, pos1), ner_type = v['sent_id'], v['global_pos'], v['type']

                        pos0 = bert_starts[pos0]
                        pos1 = bert_starts[pos1] if pos1 < len(bert_starts) else 1024

                        if (pos0, pos1) in already_exist:
                            continue

                        if pos0 >= len(pos_id):
                            continue

                        pos_id[pos0:pos1] = idx
                        ner_id[pos0:pos1] = ner2id[ner_type]
                        mention_id[pos0:pos1] = mention_idx
                        entity2mention[idx].append(mention_idx)
                        mention_idx += 1
                        already_exist.add((pos0, pos1))
                replace_i = 0
                idx = len(entity_list)
                if entity2mention[idx] == []:
                    entity2mention[idx].append(mention_idx)
                    while mention_id[replace_i] != 0:
                        replace_i += 1
                    mention_id[replace_i] = mention_idx
                    pos_id[replace_i] = idx
                    ner_id[replace_i] = ner2id[vertex[0]['type']]
                    mention_idx += 1

                new_Ls = [0]
                for ii in range(1, len(Ls)):
                    new_Ls.append(bert_starts[Ls[ii]] if Ls[ii] < len(bert_starts) else len(bert_subwords))
                Ls = new_Ls

                # construct graph
                graph = self.create_graph(Ls, mention_id, pos_id, entity2mention)

                # construct entity graph & path
                entity_graph, path = self.create_entity_graph(Ls, pos_id, entity2mention)

                assert pos_id.max() == len(entity_list)
                assert mention_id.max() == graph.number_of_nodes() - 1

                overlap = doc.get('overlap_entity_pair', [])
                new_overlap = [tuple(item) for item in overlap]

                self.data.append({
                    'title': title,
                    'entities': entity_list,
                    'labels': new_labels,
                    'na_triple': na_triple,
                    'word_id': word_id,
                    'pos_id': pos_id,
                    'ner_id': ner_id,
                    'mention_id': mention_id,
                    'entity2mention': entity2mention,
                    'graph': graph,
                    'entity_graph': entity_graph,
                    'path': path,
                    'overlap': new_overlap
                })

            # save data
            with open(file=save_file, mode='wb') as fw:
                pickle.dump({'data': self.data, 'intrain_set': self.instance_in_train}, fw)
            print('finish reading {} and save preprocessed data to {}.'.format(src_file, save_file))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def process_xml(self, data_path, index, label2idx):
        texts = []  # [doc_num], list of list of str
        triggers = []
        labels = []  # [doc_num], list
        ids = []  # [doc_num], list

        tree = ET.parse(data_path)
        root = tree.getroot()
        for document_set in root:
            for i in index:
                document = document_set[i]
                # id
                ids.append(document.attrib['id'])

                # label
                label = document.attrib['document_level_value']
                labels.append(label2idx[label])

                # text
                doc = ''
                for sentence in document:
                    if len(sentence) > 0:
                        sent = ''
                        for text in sentence.itertext():
                            sent += text
                        sent = sent.replace('-EOP- ', '').lower()
                        doc += sent + ' '

                texts.append(doc.strip())
        return texts, labels, ids

    def create_graph(self, Ls, mention_id, entity_id, entity2mention):

        d = defaultdict(list)

        # add intra edges
        for _, mentions in entity2mention.items():
            for i in range(len(mentions)):
                for j in range(i + 1, len(mentions)):
                    d[('node', 'intra', 'node')].append((mentions[i], mentions[j]))
                    d[('node', 'intra', 'node')].append((mentions[j], mentions[i]))

        if d[('node', 'intra', 'node')] == []:
            d[('node', 'intra', 'node')].append((entity2mention[1][0], 0))

        for i in range(1, len(Ls)):
            tmp = dict()
            for j in range(Ls[i - 1], Ls[i]):
                if mention_id[j] != 0:
                    tmp[mention_id[j]] = entity_id[j]
            mention_entity_info = [(k, v) for k, v in tmp.items()]

            # add self-loop & to-globle-node edges
            for m in range(len(mention_entity_info)):
                # self-loop
                # d[('node', 'loop', 'node')].append((mention_entity_info[m][0], mention_entity_info[m][0]))

                # to global node
                d[('node', 'global', 'node')].append((mention_entity_info[m][0], 0))
                d[('node', 'global', 'node')].append((0, mention_entity_info[m][0]))

            # add inter edges
            for m in range(len(mention_entity_info)):
                for n in range(m + 1, len(mention_entity_info)):
                    if mention_entity_info[m][1] != mention_entity_info[n][1]:
                        # inter edge
                        d[('node', 'inter', 'node')].append((mention_entity_info[m][0], mention_entity_info[n][0]))
                        d[('node', 'inter', 'node')].append((mention_entity_info[n][0], mention_entity_info[m][0]))

        # add self-loop for global node
        # d[('node', 'loop', 'node')].append((0, 0))
        if d[('node', 'inter', 'node')] == []:
            d[('node', 'inter', 'node')].append((entity2mention[1][0], 0))

        graph = dgl.heterograph(d)

        return graph

    def create_entity_graph(self, Ls, entity_id, entity2mention):

        graph = dgl.DGLGraph()
        graph.add_nodes(entity_id.max())

        d = defaultdict(set)

        for i in range(1, len(Ls)):
            tmp = set()
            for j in range(Ls[i - 1], Ls[i]):
                if entity_id[j] != 0:
                    tmp.add(entity_id[j])
            tmp = list(tmp)
            for ii in range(len(tmp)):
                for jj in range(ii + 1, len(tmp)):
                    d[tmp[ii] - 1].add(tmp[jj] - 1)
                    d[tmp[jj] - 1].add(tmp[ii] - 1)
        a = []
        b = []
        for k, v in d.items():
            for vv in v:
                a.append(k)
                b.append(vv)
        graph.add_edges(a, b)

        path = dict()
        for i in range(0, graph.number_of_nodes()):
            for j in range(i + 1, graph.number_of_nodes()):
                a = set(graph.successors(i).numpy())
                b = set(graph.successors(j).numpy())
                c = [val + 1 for val in list(a & b)]
                path[(i + 1, j + 1)] = c

        return graph, path


class Bert():
    MASK = '[MASK]'
    CLS = "[CLS]"
    SEP = "[SEP]"

    def __init__(self, model_class, model_name, model_path=None):
        super().__init__()
        self.model_name = model_name
        print(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.max_len = 512

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
        subwords = [self.CLS] + list(self.flatten(subwords))[:509] + [self.SEP]
        token_start_idxs = 1 + np.cumsum([0] + subword_lengths[:-1])
        token_start_idxs[token_start_idxs > 509] = 512
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
        return subword_ids.numpy(), token_start_idxs, subwords

    def segment_ids(self, segment1_len, segment2_len):
        ids = [0] * segment1_len + [1] * segment2_len
        return torch.tensor([ids])



if __name__ == '__main__':
    data_dir = '../data/'
    rel2id = json.load(open(os.path.join(data_dir, 'rel2id.json'), "r"))
    word2id = json.load(open(os.path.join(data_dir, 'word2id.json'), "r"))
    ner2id = json.load(open(os.path.join(data_dir, 'ner2id.json'), "r"))
    train_set = BERTDGLREDataset('../data/train_annotated.json', '../data/train.pkl', word2id, ner2id, rel2id,
                                 dataset_type='train')