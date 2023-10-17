import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaPreTrainedModel, RobertaModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from .gnn_layer import GraphAttentionLayer
from typing import List, Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

class Attention(nn.Module):
    def __init__(self,embed_dim,hidden_dim=None,out_dim=None,n_head=1,score_function='dot_product',dropout=0):
        super(Attention,self).__init__()
        if hidden_dim is None:
            hidden_dim = hidden_dim // n_head
        if out_dim is None:
            out_dim = hidden_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_k = nn.Linear(embed_dim,n_head*hidden_dim)
        self.w_q = nn.Linear(embed_dim,n_head*hidden_dim)
        self.proj = nn.Linear(n_head*hidden_dim,out_dim)
        self.dropout = dropout

        if self.score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim*2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim,hidden_dim))
        else:
            self.register_parameter('weight',None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv,stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:
            q = torch.unsqueeze(q,dim=1)
        if len(k.shape) == 2:
            k = torch.unsqueeze(k,dim=1)
        mb_size = k.shape[0]
        k_len = k.shape[1]
        q_len = q.shape[1]
        # k: (?, k_len, embed_dim,)
        # q: (?, q_len, embed_dim,)
        # kx: (n_head*?, k_len, hidden_dim)
        # qx: (n_head*?, q_len, hidden_dim)
        # score: (n_head*?, q_len, k_len,)
        # output: (?, q_len, out_dim,)
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)
        if self.score_function == 'dot_product':
            kt = kx.permute(0,2,1)
            score = torch.bmm(qx,kt)
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)  # (n_head*?, q_len, k_len, hidden_dim*2)
            # kq = torch.unsqueeze(kx, dim=1) + torch.unsqueeze(qx, dim=2)
            score = F.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        score = F.softmax(score,dim=-1)
        output = torch.bmm(score, kx)  # (n_head*?, q_len, hidden_dim)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)  # (?, q_len, n_head*hidden_dim)
        output = self.proj(output)  # (?, q_len, out_dim)
        output = self.dropout(output)
        return output, score

class lstm(nn.Module):
    def __init__(self,input_size,hidden_size,bidirectional=True):
        super(lstm,self).__init__()
        self.input_size = input_size
        if bidirectional:
            self.hidden_size = hidden_size // 2
        else:
            self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.LNx = nn.LayerNorm(4*self.hidden_size)
        self.LNh = nn.LayerNorm(4*self.hidden_size)
        self.LNc = nn.LayerNorm(self.hidden_size)
        self.Wx = nn.Linear(in_features=self.input_size,out_features=4*self.hidden_size,bias=True)
        self.Wh = nn.Linear(in_features=self.hidden_size,out_features=4*self.hidden_size,bias=True)

    def forward(self,x):
        def recurrence(xt,hidden): #enhanced with layer norm
            # input: input to the current cell
            htm1,ctm1 = hidden
            gates = self.LNx(self.Wx(xt)) + self.LNh(self.Wh(htm1))
            it, ft, gt, ot = gates.chunk(4,1)
            it = torch.sigmoid(it)
            ft = torch.sigmoid(ft)
            gt = torch.tanh(gt)
            ot = torch.sigmoid(ot)
            ct = (ft*ctm1) + (it*gt)
            ht = ot*torch.tanh(self.LNc(ct))
            return ht,ct
        output = []
        steps = range(x.size(1))
        hidden = self.init_hidden(x.size(0))
        inputs = x.transpose(0,1)
        for t in steps:
            hidden = recurrence(inputs[t],hidden)
            output.append(hidden[0])
        output = torch.stack(output,0).transpose(0,1)
        if self.bidirectional:
            hidden_b = self.init_hidden(x.size(0))
            output_b = []
            for t in steps[::-1]:
                hidden_b = recurrence(inputs[t],hidden_b)
                output_b.append(hidden_b[0])
            output_b = output_b[::-1]
            output_b = torch.stack(output_b,0).transpose(0,1)
            output = torch.cat([output,output_b],dim=-1)
        return output
    def init_hidden(self,bs):
        h_0 = torch.zeros(bs,self.hidden_size).cuda()
        c_0 = torch.zeros(bs,self.hidden_size).cuda()
        return h_0,c_0

class GraphNN(nn.Module):
    def __init__(self, configs):
        super(GraphNN, self).__init__()
        in_dim = 768
        configs.gnn_dims = '192'
        configs.att_heads = '4'
        configs.dp = 0.1
        self.gnn_dims = [in_dim] + [int(dim) for dim in configs.gnn_dims.strip().split(',')]

        self.gnn_layers = len(self.gnn_dims) - 1
        self.att_heads = [int(att_head) for att_head in configs.att_heads.strip().split(',')]
        self.gnn_layer_stack = nn.ModuleList()
        for i in range(self.gnn_layers):
            in_dim = self.gnn_dims[i] * self.att_heads[i - 1] if i != 0 else self.gnn_dims[i]
            self.gnn_layer_stack.append(
                GraphAttentionLayer(self.att_heads[i], in_dim, self.gnn_dims[i + 1], configs.dp)
            )

    def forward(self, doc_sents_h, doc_len, adj):
        batch, max_doc_len, _ = doc_sents_h.size()
        assert max(doc_len) == max_doc_len

        for i, gnn_layer in enumerate(self.gnn_layer_stack):
            doc_sents_h = gnn_layer(doc_sents_h, adj)

        return doc_sents_h



_CHECKPOINT_FOR_DOC = "roberta-base"
_CONFIG_FOR_DOC = "RobertaConfig"
_TOKENIZER_FOR_DOC = "RobertaTokenizer"

ROBERTA_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`RobertaConfig`]): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

ROBERTA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`RobertaTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

@add_start_docstrings(
    """
    Roberta Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    ROBERTA_START_DOCSTRING,
)
class RobertaForQuestionAnswering(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=True)
        self.gnn = GraphNN(config)
        self.dropout = nn.Dropout(0.1)
        self.attn = Attention(
            embed_dim = 768,
            hidden_dim = 768,
            n_head = 2,
            score_function='mlp',
            dropout = self.dropout
        )
        self.attn_1 = Attention(
            embed_dim = 768,
            hidden_dim = 768,
            n_head = 2,
            score_function='mlp',
            dropout = self.dropout
        )
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.tagger = lstm(
            input_size = config.hidden_size,
            hidden_size = config.hidden_size,
            bidirectional = True
        )
        # Initialize weights and apply final processing
        self.post_init()
    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint="deepset/roberta-base-squad2",
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="' puppet'",
        expected_loss=0.86,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        all_token_idx: Optional[torch.LongTensor] =None,
        all_segment_idx: Optional[torch.LongTensor] =None,
        all_clause_idx: Optional[torch.LongTensor] =None,
        all_conv_len: Optional[torch.LongTensor] =None,
        adj_b: Optional[torch.LongTensor] =None,
        all_utterance_input: Optional[torch.LongTensor] =None,
        all_q_query_input: Optional[torch.LongTensor] =None,
        all_q_query_mask: Optional[torch.LongTensor] =None,
        all_q_query_token_type: Optional[torch.LongTensor] =None
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output_0 = outputs[0]

        # sequence_output_0_lstm = self.tagger(sequence_output_0)
        # outputs_query = self.roberta(input_ids=all_q_query_input,attention_mask=all_q_query_mask,token_type_ids=all_q_query_token_type)
        # outputs_query = outputs_query[1]

        # u_num = adj_b.shape[1] #every utterance code and gcn
        # b_size = adj_b.shape[0] 
        # res = torch.empty(size=(b_size,u_num, 768)).cuda()
        # ##all_conv_len[i]
        # for i in range(all_utterance_input.shape[0]):
        #     u_temp = all_utterance_input[i][0:u_num]
        #     utterance_outputs_temp = self.roberta(input_ids=u_temp)[1]
        #     utterance_outputs_temp = utterance_outputs_temp.unsqueeze(0)
        #     u_gnn_h = self.gnn(utterance_outputs_temp,[u_num],adj_b[i][0:u_num,0:u_num].unsqueeze(0))
        #     ##q_u_att = q_u_att.squeeze(0)
        #     res[i] = u_gnn_h
        # ## q_u_att,q_u_score = self.attn(res,outputs_query.unsqueeze(1).expand(b_size,u_num,768))
        # sequence_output,score = self.attn_1(res, sequence_output_0)

        bert_output_conv = self.roberta(input_ids=all_token_idx,
                                        attention_mask=all_segment_idx)
        conv_l = max(all_conv_len).item()
        doc_sents_h = self.batched_index_select(bert_output_conv, all_clause_idx[:,0:conv_l])
        # doc_h_lstm = self.tagger(doc_sents_h)
        doc_sents_h = self.gnn(doc_sents_h, all_conv_len, adj_b)
        sequence_output,score = self.attn(doc_sents_h, sequence_output_0)

        # u_num = adj_b.shape[1]
        # b_size = adj_b.shape[0]
        # res = torch.empty(size=(b_size,u_num, 768)).cuda()
        # ##all_conv_len[i]
        # for i in range(all_utterance_input.shape[0]):
        #     u_temp = all_utterance_input[i][0:u_num]
        #     utterance_outputs_temp = self.roberta(input_ids=u_temp)[1]
        #     utterance_outputs_temp = utterance_outputs_temp.unsqueeze(0)
        #     u_gnn_h = self.gnn(utterance_outputs_temp,[u_num],adj_b[i][0:u_num,0:u_num].unsqueeze(0))
        #     ##q_u_att = q_u_att.squeeze(0)
        #     res[i] = u_gnn_h
        # ## q_u_att,q_u_score = self.attn(res,outputs_query.unsqueeze(1).expand(b_size,u_num,768))
        # sequence_output,score = self.attn_1(res, sequence_output_0)
        logits = self.qa_outputs(sequence_output) # sequence_output_0
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


    def batched_index_select(self, bert_output, bert_clause_b):
        hidden_state = bert_output[0]
        dummy = bert_clause_b.unsqueeze(2).expand(bert_clause_b.size(0), bert_clause_b.size(1), hidden_state.size(2))
        doc_sents_h = hidden_state.gather(1, dummy)
        return doc_sents_h