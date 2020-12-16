import dgl
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn
from transformers import BertModel


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

        self.gcn_dim = config.gcn_dim
        assert self.gcn_dim == config.bert_hid_size + config.entity_id_size + config.entity_type_size

        rel_name_lists = ['global', 'ss', 'st']
        self.GCN_layers = nn.ModuleList([RelGraphConvLayer(self.gcn_dim, self.gcn_dim, rel_name_lists,
                                                           num_bases=len(rel_name_lists), activation=self.activation,
                                                           self_loop=True, dropout=self.config.dropout)
                                         for i in range(config.gcn_layers)])

        self.dropout = nn.Dropout(self.config.dropout)
        self.predict = nn.Sequential(
            nn.Linear(self.bank_size * 5 + self.gcn_dim * 4, self.bank_size * 2),
            self.activation,
            self.dropout,
            nn.Linear(self.bank_size * 2, config.relation_nums),
        )

    def forward(self, **params):
        '''
        words: [batch_size, max_length]
        src_lengths: [batchs_size]
        mask: [batch_size, max_length]
        entity_type: [batch_size, max_length]
        entity_id: [batch_size, max_length]
        mention_id: [batch_size, max_length]
        distance: [batch_size, max_length]
        entity2mention_table: list of [local_entity_num, local_mention_num]
        graphs: list of DGLHeteroGraph
        h_t_pairs: [batch_size, h_t_limit, 2]
        ht_pair_distance: [batch_size, h_t_limit]
        '''
        words = params['words']
        mask = params['mask']
        bsz, slen = words.size()

        encoder_outputs, sentence_cls = self.bert(input_ids=words, attention_mask=mask)
        # encoder_outputs[mask == 0] = 0

        graphs = params['graphs']

        trigger_id = params['trigger_id']
        sentence_id = params['sentence_id']
        features = None

        for i in range(len(graphs)):
            encoder_output = encoder_outputs[i]  # [slen, bert_hid]
            trigger_num = torch.max(trigger_id[i])
            trigger_index = (torch.arange(trigger_num) + 1).unsqueeze(1).expand(-1, slen).to(self.device)  # [
            # mention_num, slen]
            triggers = trigger_id[i].unsqueeze(0).expand(trigger_num, -1)  # [mention_num, slen]
            trigger_select_metrix = (trigger_index == triggers).float()  # [mention_num, slen]
            # average word -> mention
            trigger_word_total_numbers = torch.sum(trigger_select_metrix, dim=-1).unsqueeze(-1).expand(-1, slen)  # [
            # mention_num, slen]
            trigger_select_metrix = torch.where(trigger_word_total_numbers > 0, trigger_select_metrix /
                                        trigger_word_total_numbers,
                                        trigger_select_metrix)

            trigger_x = torch.mm(trigger_select_metrix, encoder_output)  # [mention_num, bert_hid]

            sentence_num = torch.max(sentence_id[i])
            sentence_index = (torch.arange(sentence_num) + 1).unsqueeze(1).expand(-1, slen).to(self.device)  # [
            # mention_num, slen]
            sentences = sentence_id[i].unsqueeze(0).expand(sentence_num, -1)  # [mention_num, slen]
            sentence_select_metrix = (sentence_index == sentences).float()  # [mention_num, slen]
            # average word -> mention
            sentence_word_total_numbers = torch.sum(sentence_select_metrix, dim=-1).unsqueeze(-1).expand(-1, slen)  # [
            # mention_num, slen]
            sentence_select_metrix = torch.where(sentence_word_total_numbers > 0,
                                                 sentence_select_metrix / sentence_word_total_numbers,
                                                 sentence_select_metrix)

            sentence_x = torch.mm(sentence_select_metrix, encoder_output)  # [mention_num, bert_hid]
            x = torch.cat((sentence_cls[i].unsqueeze(0), sentence_x), dim=0)
            x = torch.cat((x, trigger_x), dim=0)

            if features is None:
                features = x
            else:
                features = torch.cat((features, x), dim=0)

        graph_big = dgl.batch(graphs)
        output_features = [features]

        for GCN_layer in self.GCN_layers:
            features = GCN_layer(graph_big, {"node": features})["node"]  # [total_mention_nums, gcn_dim]
            output_features.append(features)

        output_feature = torch.cat(output_features, dim=-1)

        graphs = dgl.unbatch(graph_big)

        idx = 0
        document_feature = output_feature[idx].unsuqeeze(0)
        for i in range(len(graphs) - 1):
            idx += graphs[i].number_of_nodes('node')
            document_feature = torch.cat((document_feature, output_feature[idx].unsqueeze(0)), dim=0)

        # classification
        predictions = self.predict(document_feature)
        return predictions


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
