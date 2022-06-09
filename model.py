import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from transformers.modeling_bert import BertIntermediate, BertOutput, BertAttention
from transformers.modeling_bert import BertModel, BertPreTrainedModel, BertSelfAttention

from halonet_pytorch import HaloAttention


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,  #B num_generated_triples H
        encoder_hidden_states,
        encoder_attention_mask
    ):
        self_attention_outputs = self.attention(hidden_states)
        attention_output = self_attention_outputs[0] #hidden_states.shape
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        elif encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :] # B 1 1 H
        else:
            raise ValueError(
                "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                    encoder_hidden_shape, encoder_attention_mask.shape
                )
            )
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0 #1 1 0 0 -> 0 0 -1000 -1000
        cross_attention_outputs = self.crossattention(
            hidden_states=attention_output, encoder_hidden_states=encoder_hidden_states,  encoder_attention_mask=encoder_extended_attention_mask
        )
        attention_output = cross_attention_outputs[0] #B m H
        outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output) #B m H
        outputs = (layer_output,) + outputs
        return outputs



class GRTE(BertPreTrainedModel):
    def __init__(self, config):
        super(GRTE, self).__init__(config)
        self.bert=BertModel(config=config)
        if config.fix_bert_embeddings:
            self.bert.embeddings.word_embeddings.weight.requires_grad = False
            self.bert.embeddings.position_embeddings.weight.requires_grad = False
            self.bert.embeddings.token_type_embeddings.weight.requires_grad = False

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.Lr_e1=nn.Linear(config.hidden_size,config.hidden_size)
        self.Lr_e2=nn.Linear(config.hidden_size,config.hidden_size)

        self.elu=nn.ELU()
        self.Cr = nn.Linear(config.hidden_size, config.num_p*config.num_label)

        self.Lr_e1_rev=nn.Linear(config.num_p*config.num_label,config.hidden_size)
        self.Lr_e2_rev=nn.Linear(config.num_p*config.num_label,config.hidden_size)

        self.rounds=config.rounds

        self.e_layer=DecoderLayer(config)

        torch.nn.init.orthogonal_(self.Lr_e1.weight, gain=1)
        torch.nn.init.orthogonal_(self.Lr_e2.weight, gain=1)
        torch.nn.init.orthogonal_(self.Cr.weight, gain=1)
        torch.nn.init.orthogonal_(self.Lr_e1_rev.weight, gain=1)
        torch.nn.init.orthogonal_(self.Lr_e2_rev.weight, gain=1)

    def forward(self, token_ids, mask_token_ids):
        '''
        :param token_ids:
        :param token_type_ids:
        :param mask_token_ids:
        :param s_loc:
        :return: s_pred: [batch,seq,2]
        op_pred: [batch,seq,p,2]
        '''

        embed=self.get_embed(token_ids, mask_token_ids)
        #embed:BLH
        L=embed.shape[1]

        e1 = self.Lr_e1(embed) # BLL H
        e2 = self.Lr_e2(embed)

        for i in range(self.rounds):
            h = self.elu(e1.unsqueeze(2).repeat(1, 1, L, 1) * e2.unsqueeze(1).repeat(1, L, 1, 1))  # BLL H
            B, L = h.shape[0], h.shape[1]

            table_logist = self.Cr(h)  # BLL RM

            if i!=self.rounds-1:

                table_e1 = table_logist.max(dim=2).values
                table_e2 = table_logist.max(dim=1).values
                e1_ = self.Lr_e1_rev(table_e1)
                e2_ = self.Lr_e2_rev(table_e2)

                e1=e1+self.e_layer(e1_,embed,mask_token_ids)[0]
                e2=e2+self.e_layer(e2_,embed,mask_token_ids)[0]

        return table_logist.reshape([B,L,L,self.config.num_p,self.config.num_label])

    def get_embed(self,token_ids, mask_token_ids):
        bert_out = self.bert(input_ids=token_ids.long(), attention_mask=mask_token_ids.long())
        embed=bert_out[0]
        embed=self.dropout(embed)
        return embed
    
    
class GRTE_CNN(GRTE):
    def __init__(self, config):
        super(GRTE_CNN, self).__init__(config)
        
        self.cnn =nn.Sequential(
            nn.Conv2d(config.num_p*config.num_label, 15, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(15, 15, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(15, 15, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(15, 15, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(15, 15, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(15, 15, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(15, 15, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(15, config.num_p*config.num_label, kernel_size=3, padding=1),
        )
        
    def forward(self, token_ids, mask_token_ids):

        embed=self.get_embed(token_ids, mask_token_ids)
        #embed:BLH
        L=embed.shape[1]

        e1 = self.Lr_e1(embed) # BLL H
        e2 = self.Lr_e2(embed)
        table_logist = self.elu(e1.unsqueeze(2).repeat(1, 1, L, 1) * e2.unsqueeze(1).repeat(1, L, 1, 1))  # BLL H
        B, L = table_logist.shape[0], table_logist.shape[1]
        table_logist = self.Cr(table_logist)  # BLL RM

        for i in range(self.rounds):
            table_logist_ = table_logist.permute(0, 3, 1, 2) # B RM LL
            table_logist_ = self.cnn(table_logist_) # B RM LL
            table_logist_ = table_logist_.permute(0, 2, 3, 1) # BLL RM                
            table_logist = table_logist+table_logist_

        return table_logist.reshape([B,L,L,self.config.num_p,self.config.num_label])
    
    
class GRTE_SASA(GRTE):
    def __init__(self, config):
        super(GRTE_SASA, self).__init__(config)
        
        #self.conv = nn.Sequential(
        #    nn.Linear(config.num_p*config.num_label, 100),
        #    nn.PReLU(),
        #    AttentionConv(100, 100, kernel_size=3, padding=1),
        #    nn.Linear(100, config.num_p*config.num_label)
        #)
        
        self.conv_in = nn.Linear(config.num_p*config.num_label, 100)
        self.conv_in_activ = nn.PReLU()
        self.conv = AttentionConv(100, 100, kernel_size=3, padding=1)
        self.conv_out = nn.Linear(100, config.num_p*config.num_label)
        
        # self.conv = AttentionConv(config.num_p*config.num_label, config.num_p*config.num_label, kernel_size=3, padding=1)
        
    def forward(self, token_ids, mask_token_ids):

        embed=self.get_embed(token_ids, mask_token_ids)
        #embed:BLH
        L=embed.shape[1]

        e1 = self.Lr_e1(embed) # BLL H
        e2 = self.Lr_e2(embed)
        table_logist = self.elu(e1.unsqueeze(2).repeat(1, 1, L, 1) * e2.unsqueeze(1).repeat(1, L, 1, 1))  # BLL H
        B, L = table_logist.shape[0], table_logist.shape[1]
        table_logist = self.Cr(table_logist)  # BLL RM

        for i in range(self.rounds):
            if i!=self.rounds-1:
                table_logist_ = self.conv_in_activ(self.conv_in(table_logist)) # B LL 100
                table_logist_ = table_logist_.permute(0, 3, 1, 2) # B 100 LL
                table_logist_ = self.conv(table_logist_) # B 100 LL
                table_logist_ = table_logist_.permute(0, 2, 3, 1) # BLL 100
                table_logist_ = self.conv_out(table_logist_) # BLL RM
                table_logist = table_logist+table_logist_

        return table_logist.reshape([B,L,L,self.config.num_p,self.config.num_label])
    
    
class GRTE_HALO(GRTE):
    def __init__(self, config):
        super(GRTE_HALO, self).__init__(config)
        
        self.conv = HaloAttention(dim=128, block_size=2, halo_size=4, dim_head=4, heads=4)
        self.conv_in = nn.Linear(config.num_p*config.num_label, 128)
        self.conv_in_activ = nn.PReLU()
        self.conv_out = nn.Linear(128, config.num_p*config.num_label)
        
    def forward(self, token_ids, mask_token_ids):

        embed=self.get_embed(token_ids, mask_token_ids)
        #embed:BLH
        L=embed.shape[1]

        e1 = self.Lr_e1(embed) # BLL H
        e2 = self.Lr_e2(embed)
        table_logist = self.elu(e1.unsqueeze(2).repeat(1, 1, L, 1) * e2.unsqueeze(1).repeat(1, L, 1, 1))  # BLL H
        B, L = table_logist.shape[0], table_logist.shape[1]
        table_logist = self.Cr(table_logist)  # BLL RM

        for i in range(self.rounds):
            if i!=self.rounds-1:
                table_logist_ = self.conv_in_activ(self.conv_in(table_logist)) # B LL 512
                table_logist_ = table_logist_.view(B,-1,L,L) # B 512 LL
                table_logist_ = self.conv(table_logist_) # B 512 LL
                table_logist_ = table_logist_.view(B, L, L, -1) # BLL 512
                table_logist_ = self.conv_out(table_logist_) # BLL RM
                table_logist = table_logist+table_logist_
    
        return table_logist.reshape([B,L,L,self.config.num_p,self.config.num_label])

        # embed=self.get_embed(token_ids, mask_token_ids)
        # #embed:BLH
        # B, L = embed.shape[0],embed.shape[1]
        # # L=embed.shape[1]

        # e1 = self.Lr_e1(embed) # BLL H
        # e2 = self.Lr_e2(embed)
        # table_logist = self.elu(e1.unsqueeze(2).repeat(1, 1, L, 1) * e2.unsqueeze(1).repeat(1, L, 1, 1))  # BLL H
        # table_logist = self.Cr(table_logist)
        # table_logist = table_logist.view(B,-1,L,L)
        # table_logist = self.conv(table_logist)
        # B, L = table_logist.shape[0], table_logist.shape[1]

        
        # if L//8 !=0:
        #     padded_L = 8 * (L//8 + 1)
        #     pad_len = padded_L - L
        #     head_pad_len = pad_len//2 if pad_len//2==0 else pad_len//2
        #     tail_pad_len = front_pad_len if pad_len//2==0 else pad_len//2+1
        #     table_logist = F.pad(table_logist, (0, 0, head_pad_len, tail_pad_len, head_pad_len, tail_pad_len))
        
        # table_logist = self.Cr(table_logist)  # BLL RM

        # for i in range(self.rounds):
        #     table_logist_ = table_logist.permute(0, 3, 1, 2) # B RM LL
        #     table_logist_ = self.conv(table_logist_) # B RM LL
        #     table_logist_ = table_logist_.permute(0, 2, 3, 1) # BLL RM                
        #     table_logist = table_logist+table_logist_

        # return table_logist.reshape([B,L,L,self.config.num_p,self.config.num_label])
    
    
class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(AttentionConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)