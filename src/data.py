import os
import pickle
from collections import defaultdict
import xml.etree.ElementTree as ET
from sklearn.model_selection import StratifiedKFold

import dgl
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


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


class Bert():
    MASK = '[MASK]'
    CLS = "[CLS]"
    SEP = "[SEP]"

    def __init__(self, model_name, model_path=None):
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
        subwords = [self.CLS] + list(self.flatten(subwords))[:510] + [self.SEP]
        token_start_idxs = 1 + np.cumsum([0] + subword_lengths[:-1])
        token_start_idxs[token_start_idxs > 510] = 511
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
        return subword_ids[0], mask[0], token_start_idxs, subwords

    def segment_ids(self, segment1_len, segment2_len):
        ids = [0] * segment1_len + [1] * segment2_len
        return torch.tensor([ids])


def collate(samples):
    ids, labels, words, masks, sentence_id, trigger_id, graphs = map(list, zip(*samples))
    batched_ids = tuple(ids)
    batched_labels = torch.tensor(labels)
    batched_words = torch.stack(words)
    batched_masks = torch.stack(masks)
    batched_sentence_id = torch.stack(sentence_id)
    batched_trigger_id = torch.stack(trigger_id)
    batched_graph = dgl.batch(graphs)
    return batched_ids, batched_labels, batched_words, batched_masks, \
           batched_sentence_id, batched_trigger_id, batched_graph


def get_data(opt, label2idx, index):
    trainset = BERTDGLREDataset(opt.data_path, opt.data_save_path, label2idx,
                                index, dataset_type='train', bert_path=opt.bert_path)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size,
                             shuffle=True, num_workers=opt.num_workers,
                             collate_fn=collate)
    testset = BERTDGLREDataset(opt.data_path, opt.data_save_path, label2idx,
                                index, dataset_type='test', bert_path=opt.bert_path)
    testloader = DataLoader(testset, batch_size=opt.test_batch_size,
                            shuffle=False, num_workers=opt.num_workers,
                            collate_fn=collate)
    return trainloader, testloader


if __name__ == '__main__':
    index, label2idx = k_fold_split("../data/dlef_corpus/english.xml", 5)
    train_set = BERTDGLREDataset('../data/dlef_corpus/english.xml', '../data/train.pkl', label2idx, index[0],
                                 dataset_type='train', bert_path="../../data/bert-base-uncased")
    a, b, c, d, e, f, g = train_set.__getitem__(0)
    dataloader = DataLoader(train_set, batch_size=2, shuffle=False, collate_fn=collate)
    for a, b, c, d, e, f, g in dataloader:
        g_unbatch = dgl.unbatch(g)
        print("hello")
    print("end")
