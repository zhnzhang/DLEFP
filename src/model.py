import dgl
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

import math
from entmax import entmax15


class GAIN_BERT(nn.Module):
    def __init__(self, config, num_classes=5):
        super(GAIN_BERT, self).__init__()
        self.config = config
        self.device = 'cuda' if config.gpu else 'cpu'
        if config.activation == 'tanh':
            self.activation = nn.Tanh()
        elif config.activation == 'relu':
            self.activation = nn.ReLU()
        else:
            assert 1 == 2, "you should provide activation function."

        self.bert = BertModel.from_pretrained(config.bert_path)
        if config.bert_fix:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.attn = MultiHeadAttention(config.n_heads, config.bert_hid_size)
        self.sent_gcn = GCN(config.bert_hid_size, config.bert_hid_size, config.bert_hid_size)

        # self.type_embedding = nn.Embedding(num_embeddings=3, embedding_dim=config.type_embed_dim)

        self.gcn_in_dim = config.bert_hid_size
        self.gcn_hid_dim = config.gcn_hid_dim
        self.gcn_out_dim = config.gcn_out_dim
        self.dropout = config.dropout

        rel_name_lists = ['neighbor', 'global', 'td', 'ts']
        self.GCN_layers = nn.ModuleList()
        self.GCN_layers.append(RelGraphConvLayer(self.gcn_in_dim, self.gcn_hid_dim, rel_name_lists,
                                                 num_bases=len(rel_name_lists), activation=self.activation,
                                                 self_loop=True, dropout=self.dropout))
        self.GCN_layers.append(RelGraphConvLayer(self.gcn_hid_dim, self.gcn_out_dim, rel_name_lists,
                                                 num_bases=len(rel_name_lists), activation=self.activation,
                                                 self_loop=True, dropout=self.dropout))

        self.bank_size = self.gcn_in_dim + self.gcn_hid_dim + self.gcn_out_dim
        # self.linear_dim = config.linear_dim
        '''self.predict = nn.Sequential(
            nn.Linear(self.bank_size, self.linear_dim),
            self.activation,
            nn.Dropout(self.dropout),
            nn.Linear(self.linear_dim, num_classes),
        )'''
        self.predict = nn.Linear(self.bank_size, num_classes)
        self.trigger_predict = nn.Linear(self.bank_size, num_classes)

    def forward(self, **params):
        doc_ids = params['doc_ids']  # [bsz, doc_len]
        doc_mask = params['doc_mask']  # [bsz, doc_len]
        bsz = doc_ids.size()[0]
        _, document_cls = self.bert(input_ids=doc_ids, attention_mask=doc_mask)  # [bsz, bert_dim]

        sent_ids = params['sent_ids']  # [bsz * seq_num, seq_len]
        sent_mask = params['sent_mask']  # [bsz * seq_num, seq_len]
        sentence_embed, sentence_cls = self.bert(input_ids=sent_ids, attention_mask=sent_mask)
        _, seq_len, bert_dim = sentence_embed.shape
        # sentence_embed: [bsz * seq_num, seq_len, bert_dim]
        # sentence_cls: [bsz * seq_num, bert_dim]

        # sentence graph
        s_idx = params['sent_idx']  # bsz * [seq_num, word_num, seq_len]
        s_dep_adj = params['sent_dep_adj']  # bsz * [seq_num, word_num, word_num]
        t_sid = params['trigger_sid']  # bsz * [trigger_num]
        t_index = params['trigger_index']  # bsz * [trigger_num]

        batched_graph = params['graph']
        graphs = dgl.unbatch(batched_graph)

        assert len(graphs) == bsz, "batch size inconsistent"

        split_sizes = []
        for i in range(bsz):
            sentence_num = s_idx[i].shape[0]
            split_sizes.append(sentence_num)
        feature_list = list(torch.split(sentence_cls, split_sizes, dim=0))
        sent = list(torch.split(sentence_embed, split_sizes, dim=0))  # bsz * [seq_num, seq_len, bert_dim]

        for i in range(bsz):
            feature_list[i] = torch.cat((document_cls[i].unsqueeze(0), feature_list[i]), dim=0)

            # trigger_features
            # sentence graph
            word_embed = torch.sum(s_idx[i].unsqueeze(-1) * sent[i].unsqueeze(1), dim=2)  # [seq_num, word_num,
            # bert_dim]

            # latent adjacent matrix
            key_padding_mask = torch.sum(s_idx[i], dim=-1)  # [seq_num, word_num]
            s_lat_adj = self.attn(word_embed, word_embed, mask=key_padding_mask)

            x = self.sent_gcn(word_embed, s_dep_adj[i], s_lat_adj)  # [seq_num, word_num, s_bank_size]

            t_feats = []
            trigger_num = t_sid[i].shape[0]
            for j in range(trigger_num):
                t_feats.append(x[t_sid[i][j]][t_index[i][j]])
            t_feats = torch.stack(t_feats, dim=0)
            feature_list[i] = torch.cat((feature_list[i], t_feats), dim=0)

        features = torch.cat(feature_list, dim=0)
        assert features.size()[0] == batched_graph.number_of_nodes('node'), "number of nodes inconsistent"
        output_features = [features]

        for GCN_layer in self.GCN_layers:
            features = GCN_layer(batched_graph, {"node": features})["node"]  # [total_node_nums, gcn_dim]
            output_features.append(features)

        output_feature = torch.cat(output_features, dim=-1)
        assert output_feature.size()[0] == batched_graph.number_of_nodes('node'), "number of nodes inconsistent"

        idx = 0
        document_features = []
        trigger_features = []
        for i in range(len(graphs)):
            document_features.append(output_feature[idx])

            trigger_start = idx + 1 + split_sizes[i]
            idx += graphs[i].number_of_nodes('node')

            trigger_features.append(output_feature[trigger_start:idx])
        document_feature = torch.stack(document_features, dim=0)
        trigger_feature = torch.cat(trigger_features, dim=0)

        # classification
        predictions = self.predict(document_feature)
        trigger_predictions = self.trigger_predict(trigger_feature)
        return predictions, trigger_predictions


class Attention(nn.Module):
    def __init__(self, src_size, trg_size):
        super().__init__()
        self.W = nn.Bilinear(src_size, trg_size, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src, trg, attention_mask=None):
        '''
        src: [src_size]
        trg: [middle_node, trg_size]
        '''

        score = self.W(src.unsqueeze(0).expand(trg.size(0), -1), trg)
        score = self.softmax(score)
        value = torch.mm(score.permute(1, 0), trg)

        return score.squeeze(0), value.squeeze(0)


class BiLSTM(nn.Module):
    def __init__(self, input_size, config):
        super().__init__()
        self.config = config
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=config.lstm_hidden_size,
                            num_layers=config.nlayers, batch_first=True,
                            bidirectional=True)
        self.in_dropout = nn.Dropout(config.dropout)
        self.out_dropout = nn.Dropout(config.dropout)

    def forward(self, src, src_lengths):
        '''
        src: [batch_size, slen, input_size]
        src_lengths: [batch_size]
        '''

        self.lstm.flatten_parameters()
        bsz, slen, input_size = src.size()

        src = self.in_dropout(src)

        new_src_lengths, sort_index = torch.sort(src_lengths, dim=-1, descending=True)
        new_src = torch.index_select(src, dim=0, index=sort_index)

        packed_src = nn.utils.rnn.pack_padded_sequence(new_src, new_src_lengths, batch_first=True, enforce_sorted=True)
        packed_outputs, (src_h_t, src_c_t) = self.lstm(packed_src)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True,
                                                      padding_value=self.config.word_pad)

        unsort_index = torch.argsort(sort_index)
        outputs = torch.index_select(outputs, dim=0, index=unsort_index)

        src_h_t = src_h_t.view(self.config.nlayers, 2, bsz, self.config.lstm_hidden_size)
        src_c_t = src_c_t.view(self.config.nlayers, 2, bsz, self.config.lstm_hidden_size)
        output_h_t = torch.cat((src_h_t[-1, 0], src_h_t[-1, 1]), dim=-1)
        output_c_t = torch.cat((src_c_t[-1, 0], src_c_t[-1, 1]), dim=-1)
        output_h_t = torch.index_select(output_h_t, dim=0, index=unsort_index)
        output_c_t = torch.index_select(output_c_t, dim=0, index=unsort_index)

        outputs = self.out_dropout(outputs)
        output_h_t = self.out_dropout(output_h_t)
        output_c_t = self.out_dropout(output_c_t)

        return outputs, (output_h_t, output_c_t)


class RelGraphConvLayer(nn.Module):
    r"""Relational graph convolution layer.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    rel_names : list[str]
        Relation names.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    weight : bool, optional
        True if a linear layer is applied after message passing. Default: True
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    """

    def __init__(self,
                 in_feat,
                 out_feat,
                 rel_names,
                 num_bases,
                 *,
                 weight=True,
                 bias=True,
                 activation=None,
                 self_loop=False,
                 dropout=0.0):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        self.conv = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feat, out_feat, norm='right', weight=False, bias=False)
            for rel in rel_names
        })

        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = dglnn.WeightBasis((in_feat, out_feat), num_bases, len(self.rel_names))
            else:
                self.weight = nn.Parameter(torch.Tensor(len(self.rel_names), in_feat, out_feat))
                nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        # bias
        if bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        """Forward computation
        Parameters
        ----------
        g : DGLHeteroGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.
        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        g = g.local_var()
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight
            wdict = {self.rel_names[i]: {'weight': w.squeeze(0)}
                     for i, w in enumerate(torch.split(weight, 1, dim=0))}
        else:
            wdict = {}
        hs = self.conv(g, inputs, mod_kwargs=wdict)

        def _apply(ntype, h):
            if self.self_loop:
                h = h + torch.matmul(inputs[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)

        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}


class RelEdgeLayer(nn.Module):
    def __init__(self,
                 node_feat,
                 edge_feat,
                 activation,
                 dropout=0.0):
        super(RelEdgeLayer, self).__init__()
        self.node_feat = node_feat
        self.edge_feat = edge_feat
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.mapping = nn.Linear(node_feat * 2, edge_feat)

    def forward(self, g, inputs):
        # g = g.local_var()

        g.ndata['h'] = inputs  # [total_mention_num, node_feat]
        g.apply_edges(lambda edges: {
            'h': self.dropout(self.activation(self.mapping(torch.cat((edges.src['h'], edges.dst['h']), dim=-1))))})
        g.ndata.pop('h')


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.linear1 = nn.Linear(out_features, out_features)
        self.linear2 = nn.Linear(out_features, out_features)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, gate_adj=None):
        # text: [bsz, word_num, in_features]
        # adj: [bsz, word_num, word_num]
        # gate_adj: [bsz, word_num, word_num]
        hidden = torch.matmul(input, self.weight)
        output = torch.matmul(adj, hidden)

        if gate_adj is not None:
            lat_output = torch.matmul(gate_adj, hidden)

            g = torch.sigmoid(self.linear1(output) + self.linear2(lat_output))
            output = g * output + (1 - g) * lat_output

        if self.bias is not None:
            output = output + self.bias
        output = F.relu(output)
        return output


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nout)

    def forward(self, x, adj, gate_adj=None):
        res = x
        x = self.gc1(x, adj, gate_adj)
        x = self.gc2(x, adj, gate_adj)
        return x + res


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0):
        "Take in model size and number of heads."
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout) if dropout != 0 else None

    def forward(self, query, key, mask=None):
        # query and value are two copies of sentence representation H
        # query: [nbatches, seq_len, d_model]
        # value: [nbatches, seq_len, d_model]
        # mask: [nbatches, seq_len, seq_len]
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query = self.w_q(query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key = self.w_k(key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        # [nbatches, h, seq_len, d_k]

        # 2) Apply attention on all the projected vectors in batch.
        # Compute 'Scaled Dot Product Attention'
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(self.d_k)  # [nbatches, h, seq_len, seq_len]
        if mask is not None:
            key_padding_mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(key_padding_mask == 0, float("-inf"))
        p_attn = entmax15(scores, dim=-1)  # [nbatches, h, seq_len, seq_len]
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)

        # 3) "Concat" using a view and apply a final linear.
        p_attn = torch.sum(p_attn, dim=1) / self.h
        return p_attn  # [nbatches, seq_len, seq_len]