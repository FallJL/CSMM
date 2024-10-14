# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import RobertaModel, RobertaConfig
from typing import List, Optional, Tuple, Union
from transformers.models.roberta.modeling_roberta import (RobertaSelfOutput, RobertaIntermediate, RobertaOutput, RobertaEncoder,
                                                          ROBERTA_INPUTS_DOCSTRING, _CHECKPOINT_FOR_DOC, _CONFIG_FOR_DOC,
                                                          apply_chunking_to_forward, add_start_docstrings_to_model_forward, add_code_sample_docstrings,
                                                          find_pruneable_heads_and_indices, prune_linear_layer,
                                                          BaseModelOutputWithPoolingAndCrossAttentions, BaseModelOutputWithPastAndCrossAttentions,
                                                          logger)
import copy
import math

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        config = RobertaConfig.from_pretrained(args.model_name_or_path)
        config.no_mome = args.no_mome
        config.fusion_layer = args.fusion_layer

        # create model
        self.encoder = NewRobertaModel.from_pretrained(args.model_name_or_path, config=config) 
        # initialize parameter
        for i, layer in enumerate(self.encoder.encoder.layer):
            if not config.no_mome:
                layer.intermediate_code.load_state_dict(layer.intermediate.state_dict())
                layer.output_code.load_state_dict(layer.output.state_dict())
            if i >= config.fusion_layer:
                layer.crossattention.load_state_dict(layer.attention.state_dict())
                if not config.no_mome:
                    layer.intermediate_cn.load_state_dict(layer.intermediate.state_dict())
                    layer.output_cn.load_state_dict(layer.output.state_dict())

        self.args = args
        self.is_distributed = args.distributed
        self.hidden_size = self.encoder.config.hidden_size
        self.distill = args.distill
        self.inbatch_contrast = args.inbatch_contrast
        self.queue_size = args.queue_size
        self.momentum = args.momentum
        self.temp = args.temp
        # self.temp = nn.Parameter(torch.ones([]) * args.temp) # temperature parameter 
        self.match_head = nn.Linear(self.hidden_size, 2) 

        # create momentum model
        self.encoder_m = NewRobertaModel.from_pretrained(args.model_name_or_path, config=config)
        self.model_pairs = [[self.encoder,self.encoder_m]]
        self.copy_params()

        # create the queue
        self.register_buffer("code_queue", torch.randn(self.hidden_size, self.queue_size))
        self.register_buffer("nl_queue", torch.randn(self.hidden_size, self.queue_size))
        self.register_buffer("idx_queue", torch.full((1,self.queue_size),-100))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        if args.match_queue: # 用以保存match queue
            self.register_buffer("code_hidden_queue", torch.randn(self.queue_size, args.code_length, self.hidden_size))
            self.register_buffer("nl_hidden_queue", torch.randn(self.queue_size, args.nl_length, self.hidden_size))
            self.register_buffer("code_mask_queue", torch.zeros(self.queue_size, args.code_length).bool())
            self.register_buffer("nl_mask_queue", torch.zeros(self.queue_size, args.nl_length).bool())

        # normalization        
        self.code_queue = nn.functional.normalize(self.code_queue, dim=0)
        self.nl_queue = nn.functional.normalize(self.nl_queue, dim=0)

    def forward(self, code_inputs=None, nl_inputs=None, alpha=0.4, idx=None, 
                no_match=False, match_queue=False, continue_fine_tune=False):     
        if continue_fine_tune:
            if code_inputs is not None:
                code_hidden = self.encoder(code_inputs,attention_mask=code_inputs.ne(1), modality_type="code")[0] 
                code_vec = (code_hidden*code_inputs.ne(1)[:,:,None]).sum(1)/code_inputs.ne(1).sum(-1)[:,None]            
                code_vec = F.normalize(code_vec,dim=-1)
                return code_vec
            else:
                nl_hidden = self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1), modality_type="nl")[0]
                nl_vec = (nl_hidden*nl_inputs.ne(1)[:,:,None]).sum(1)/nl_inputs.ne(1).sum(-1)[:,None]            
                nl_vec = F.normalize(nl_vec,dim=-1)    
                return nl_vec  

        code_hidden = self.encoder(code_inputs,attention_mask=code_inputs.ne(1), modality_type="code")[0] 
        code_vec = (code_hidden*code_inputs.ne(1)[:,:,None]).sum(1)/code_inputs.ne(1).sum(-1)[:,None]            
        code_vec = F.normalize(code_vec,dim=-1)

        nl_hidden = self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1), modality_type="nl")[0]
        nl_vec = (nl_hidden*nl_inputs.ne(1)[:,:,None]).sum(1)/nl_inputs.ne(1).sum(-1)[:,None]            
        nl_vec = F.normalize(nl_vec,dim=-1)                      

        idx = idx.view(-1,1)
        idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()],dim=1)  
        pos_idx = torch.eq(idx, idx_all).float()       
        sim_targets = pos_idx / pos_idx.sum(1,keepdim=True)    

        # 动量, 队列
        with torch.no_grad():
            self._momentum_update()

            code_hidden_m = self.encoder_m(code_inputs,attention_mask=code_inputs.ne(1), modality_type="code")[0] 
            code_vec_m = (code_hidden_m*code_inputs.ne(1)[:,:,None]).sum(1)/code_inputs.ne(1).sum(-1)[:,None] 
            code_vec_m = F.normalize(code_vec_m,dim=-1)  # [bs, hidden_size]
            code_vec_all = torch.cat([code_vec_m.t(),self.code_queue.clone().detach()],dim=1) # [hidden_size, bs + queue_len]

            nl_hidden_m = self.encoder_m(nl_inputs,attention_mask=nl_inputs.ne(1), modality_type="nl")[0] 
            nl_vec_m = (nl_hidden_m*nl_inputs.ne(1)[:,:,None]).sum(1)/nl_inputs.ne(1).sum(-1)[:,None]    
            nl_vec_m = F.normalize(nl_vec_m,dim=-1)  # [bs, hidden_size]
            nl_vec_all = torch.cat([nl_vec_m.t(),self.nl_queue.clone().detach()],dim=1) # [hidden_size, bs + queue_len]

            # Momentum Distillation
            if self.distill:               
                sim_c2n_m = code_vec_m @ nl_vec_all / self.temp 
                sim_n2c_m = nl_vec_m @ code_vec_all / self.temp   

                sim_c2n_targets = alpha * F.softmax(sim_c2n_m, dim=1) + (1 - alpha) * sim_targets
                sim_n2c_targets = alpha * F.softmax(sim_n2c_m, dim=1) + (1 - alpha) * sim_targets 

        sim_c2n = code_vec @ nl_vec_all / self.temp 
        sim_n2c = nl_vec @ code_vec_all / self.temp           

        if self.distill: # 动量蒸馏时的损失
            loss_c2n = -torch.sum(F.log_softmax(sim_c2n, dim=1)*sim_c2n_targets,dim=1).mean()
            loss_n2c = -torch.sum(F.log_softmax(sim_n2c, dim=1)*sim_n2c_targets,dim=1).mean() 
        else:
            loss_c2n = -torch.sum(F.log_softmax(sim_c2n, dim=1)*sim_targets,dim=1).mean()
            loss_n2c = -torch.sum(F.log_softmax(sim_n2c, dim=1)*sim_targets,dim=1).mean()   

        if self.inbatch_contrast:
            sim_inbatch = code_vec @ nl_vec.t()
            loss_fct = CrossEntropyLoss()
            loss_inbatch = loss_fct(sim_inbatch, torch.arange(code_vec.size(0), device=code_vec.device))
            loss_contrast = (loss_c2n+loss_n2c+loss_inbatch)/3
        else:
            # 对比损失
            loss_contrast = (loss_c2n+loss_n2c)/2

        # 出队和入队
        if not match_queue:
            self._dequeue_and_enqueue(code_vec_m, nl_vec_m, idx, is_distributed=self.is_distributed)
        
        if no_match: # 移除match，则不用算match loss了
            return loss_contrast

        # forward the positve code-nl pair
        match_hidden_pos = self.encoder(encoder_embeds = nl_hidden, 
                                        attention_mask = nl_inputs.ne(1),
                                        encoder_hidden_states = code_hidden,
                                        encoder_attention_mask = code_inputs.ne(1),      
                                        return_dict = True,
                                        modality_type = 'fusion')
        # # symmetric input
        # match_hidden_pos_s = self.encoder(encoder_embeds = code_hidden, 
        #                             attention_mask = code_inputs.ne(1),
        #                             encoder_hidden_states = nl_hidden,
        #                             encoder_attention_mask = nl_inputs.ne(1),      
        #                             return_dict = True,
        #                             modality_type = 'fusion',
        #                             )                
        with torch.no_grad(): # hard negative probability
            bs = code_inputs.size(0)
            if match_queue: # 把队列中每个向量当时的hidden_state存下来，便于后面使用
                weights_c2n = F.softmax(sim_c2n[:,:]+1e-4,dim=1) # 形状[bs, bs + queue_len], 与mask相同
                weights_n2c = F.softmax(sim_n2c[:,:]+1e-4,dim=1)
                mask = torch.eq(idx, idx.T)

                # 把queue对应的mask也cat上
                mask_cat = torch.eq(idx, self.idx_queue.clone().detach())
                # 在queue没满时，那些无意义的样本不应该被选作负样本，
                # 应该和ground truth(对角线上的元素)一样，被置为0
                no_choose_id_start = torch.where(self.idx_queue.clone().detach()[0] == -100)[0]
                if no_choose_id_start.size(0) != 0: # queue没满
                    no_choose_id_start = int(no_choose_id_start[0])
                    mask_cat[:, no_choose_id_start:] = True
                    # 如果queue满了，就不用上一步的True赋值了，所以没有else

                mask = torch.cat((mask, mask_cat), dim=1)
            
            else:          
                weights_c2n = F.softmax(sim_c2n[:,:bs]+1e-4,dim=1) # 形状[bs, bs], 与mask相同
                weights_n2c = F.softmax(sim_n2c[:,:bs]+1e-4,dim=1)
                mask = torch.eq(idx, idx.T)
            
            weights_c2n.masked_fill_(mask, 0) # 把ground truth(对角线上的元素)置为0
            weights_n2c.masked_fill_(mask, 0)

        # select a negative code for each nl
        # 对于对称输入, 这里可以重新选负样本
        match_code_hidden_neg = []
        match_code_mask_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_n2c[b], 1).item()
            if match_queue:
                if neg_idx < bs: # 这里选择的原编码器的输出，动量编码器的编码输出也可考虑
                    match_code_hidden_neg.append(code_hidden[neg_idx])
                    match_code_mask_neg.append(code_inputs[neg_idx].ne(1))
                else:
                    match_code_hidden_neg.append(self.code_hidden_queue[neg_idx-bs].clone().detach())
                    match_code_mask_neg.append(self.code_mask_queue[neg_idx-bs].clone().detach())
            else:
                match_code_hidden_neg.append(code_hidden[neg_idx])
                match_code_mask_neg.append(code_inputs[neg_idx].ne(1))
        match_code_hidden_neg = torch.stack(match_code_hidden_neg,dim=0)
        match_code_mask_neg = torch.stack(match_code_mask_neg, dim=0)   

        # select a negative nl for each code
        # 对于对称输入, 这里可以重新选负样本
        match_nl_hidden_neg = []
        match_nl_mask_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_c2n[b], 1).item()
            if match_queue:
                if neg_idx < bs:
                    match_nl_hidden_neg.append(nl_hidden[neg_idx])
                    match_nl_mask_neg.append(nl_inputs[neg_idx].ne(1))
                else:
                    match_nl_hidden_neg.append(self.nl_hidden_queue[neg_idx-bs].clone().detach())
                    match_nl_mask_neg.append(self.nl_mask_queue[neg_idx-bs].clone().detach())
            else:
                match_nl_hidden_neg.append(nl_hidden[neg_idx])
                match_nl_mask_neg.append(nl_inputs[neg_idx].ne(1))
        match_nl_hidden_neg = torch.stack(match_nl_hidden_neg, dim=0)   
        match_nl_mask_neg = torch.stack(match_nl_mask_neg, dim=0)      

        match_code_hidden_all = torch.cat([match_code_hidden_neg, code_hidden], dim=0)
        match_code_mask_all = torch.cat([match_code_mask_neg, code_inputs.ne(1)], dim=0)
        match_nl_hidden_all = torch.cat([nl_hidden, match_nl_hidden_neg], dim=0)     
        match_nl_mask_all = torch.cat([nl_inputs.ne(1), match_nl_mask_neg], dim=0)     

        output_neg = self.encoder(encoder_embeds = match_nl_hidden_all, 
                                  attention_mask = match_nl_mask_all,
                                  encoder_hidden_states = match_code_hidden_all,
                                  encoder_attention_mask = match_code_mask_all,      
                                  return_dict = True,
                                  modality_type = 'fusion')

        # # symmetric input
        # output_neg_s = self.encoder(encoder_embeds = match_code_hidden_all, 
        #                                 attention_mask = match_code_mask_all,
        #                                 encoder_hidden_states = match_nl_hidden_all,
        #                                 encoder_attention_mask = match_nl_mask_all,      
        #                                 return_dict = True,
        #                                 modality_type = 'fusion',
        #                                )                           

        match_vec = torch.cat([match_hidden_pos.last_hidden_state[:,0,:], output_neg.last_hidden_state[:,0,:]], dim=0)
        match_logits = self.match_head(match_vec)
        match_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2*bs, dtype=torch.long)],
                                 dim=0).to(code_inputs.device)
        loss_match = F.cross_entropy(match_logits, match_labels)

        # # symmetric input
        # match_vec_s = torch.cat([match_hidden_pos_s.last_hidden_state[:,0,:], output_neg_s.last_hidden_state[:,0,:]],dim=0)
        # match_logits_s = self.match_head(match_vec_s)            

        # match_labels_s = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],
        #                        dim=0).to(code_inputs.device)
        # loss_match_s = F.cross_entropy(match_logits_s, match_labels_s)

        # loss_match_all = (loss_match + loss_match_s) / 2     

        # 如果match要用到queue，则应该推迟进队出队（比如推迟到这里），待queue要用的数据处理完
        if match_queue:
            self._dequeue_and_enqueue(code_vec_m, nl_vec_m, idx, match_queue,
                                      code_hidden_m, nl_hidden_m, 
                                      code_inputs.ne(1), nl_inputs.ne(1),
                                      is_distributed=self.is_distributed)

        lambda1 = 0.4
        return loss_contrast + loss_match * lambda1
 


    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
                
    @torch.no_grad()
    def _dequeue_and_enqueue(self, code_vec, nl_vec, idx, match_queue=False,
                             code_hidden_state=None, nl_hidden_state=None, 
                             code_atts_mask=None, nl_atts_mask=None,
                             is_distributed=True):
        # gather keys before updating queue
        code_vecs = concat_all_gather(code_vec, is_distributed)
        nl_vecs = concat_all_gather(nl_vec, is_distributed)
        idxs = concat_all_gather(idx, is_distributed)
        if match_queue:
            code_hidden_states = concat_all_gather(code_hidden_state, is_distributed)
            nl_hidden_states = concat_all_gather(nl_hidden_state, is_distributed)
            code_atts_masks = concat_all_gather(code_atts_mask, is_distributed)
            nl_atts_masks = concat_all_gather(nl_atts_mask, is_distributed)

        batch_size = code_vecs.shape[0]

        ptr = int(self.queue_ptr)
        # assert self.queue_size % batch_size == 0  # for simplicity
        if batch_size == self.args.train_batch_size:

            # replace the keys at ptr (dequeue and enqueue)
            self.code_queue[:, ptr:ptr + batch_size] = code_vecs.T
            self.nl_queue[:, ptr:ptr + batch_size] = nl_vecs.T
            self.idx_queue[:, ptr:ptr + batch_size] = idxs.T
            if match_queue:
                self.code_hidden_queue[ptr:ptr + batch_size] = code_hidden_states
                self.nl_hidden_queue[ptr:ptr + batch_size] = nl_hidden_states
                self.code_mask_queue[ptr:ptr + batch_size] = code_atts_masks
                self.nl_mask_queue[ptr:ptr + batch_size] = nl_atts_masks

            ptr = (ptr + batch_size) % self.queue_size  # move pointer

            self.queue_ptr[0] = ptr   
        
@torch.no_grad()
def concat_all_gather(tensor, is_distributed):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if is_distributed == False:
        return tensor
    
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class NewRobertaModel(RobertaModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = NewRobertaEncoder(config)

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_embeds=None, # nl encoder的输出，将输入到code-nl融合层
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        modality_type = "nl"
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            # self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device
        elif encoder_embeds is not None:    
            input_shape = encoder_embeds.size()[:-1] 
            device = encoder_embeds.device
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds or encoder_embeds")

        batch_size, seq_length = input_shape
        # device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if encoder_hidden_states is not None:
            # rjl: follow original ALBEF code
            if type(encoder_hidden_states) == list:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states[0].size()
            else:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            
            if type(encoder_attention_mask) == list:
                encoder_extended_attention_mask = [self.invert_attention_mask(mask) for mask in encoder_attention_mask]
            elif encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
            else:    
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if encoder_embeds is None:
            embedding_output = self.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_key_values_length,
            )
        else:
            embedding_output = encoder_embeds

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            modality_type=modality_type
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

class NewRobertaEncoder(RobertaEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.fusion_layer = config.fusion_layer
        self.layer = nn.ModuleList([NewRobertaLayer(config, i) for i in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        modality_type = "nl"
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None

        if modality_type=='nl': 
            start_layer = 0
            output_layer = self.config.num_hidden_layers
            
        elif modality_type=='fusion':
            start_layer = self.fusion_layer
            output_layer = self.config.num_hidden_layers
            
        elif modality_type=='code':
            start_layer = 0
            output_layer = self.config.num_hidden_layers

        for i in range(start_layer, output_layer):
            layer_module = self.layer[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    modality_type
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

class NewRobertaLayer(nn.Module):
    def __init__(self, config, layer_num):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = NewRobertaAttention(config)
        self.fusion_layer = config.fusion_layer

        self.has_cross_attention = (layer_num >= self.fusion_layer)
        if self.has_cross_attention:
            self.layer_num = layer_num
            self.crossattention = NewRobertaAttention(config, is_cross_attention=True)
        self.intermediate = RobertaIntermediate(config) # 原始的intermediate就是intermediate_nl, 负责处理自然语言
        self.output = RobertaOutput(config) # output同上

        # # rjl: Text modal expert
        # self.intermediate_nl = copy.deepcopy(self.intermediate)
        # self.output_nl = copy.deepcopy(self.output)
        # rjl: Code modal expert
        self.no_mome = config.no_mome
        if not self.no_mome:
            self.intermediate_code = copy.deepcopy(self.intermediate)
            self.output_code = copy.deepcopy(self.output)
        # # rjl: Text-code modal expert
        # self.intermediate_cn = copy.deepcopy(self.intermediate)
        # self.output_cn = copy.deepcopy(self.output)
        if self.has_cross_attention and (not self.no_mome): # 有cross_attention的层才有intermediate_cn和output_cn
            self.intermediate_cn = copy.deepcopy(self.intermediate)
            self.output_cn = copy.deepcopy(self.output)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        modality_type = "nl"
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # # if decoder, the last output is tuple of self-attn cache
        # if self.is_decoder:
        #     outputs = self_attention_outputs[1:-1]
        #     present_key_value = self_attention_outputs[-1]
        # else:
        #     outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        outputs = self_attention_outputs[1:-1]
        present_key_value = self_attention_outputs[-1]

        if encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            if type(encoder_hidden_states) == list:
                cross_attention_outputs = self.crossattention(
                    attention_output,
                    attention_mask,
                    head_mask,
                    encoder_hidden_states[(self.layer_num-self.fusion_layer)%len(encoder_hidden_states)],
                    encoder_attention_mask[(self.layer_num-self.fusion_layer)%len(encoder_hidden_states)],
                    output_attentions=output_attentions,
                )    
                attention_output = cross_attention_outputs[0]
                outputs = outputs + cross_attention_outputs[1:-1]

            else:
                cross_attention_outputs = self.crossattention(
                    attention_output,
                    attention_mask,
                    head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions=output_attentions,
                )
                attention_output = cross_attention_outputs[0]
                outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

        # rjl: Select different FFNs according to different input modalities
        if modality_type == "nl" or self.no_mome:
            layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
            )
            # layer_output = apply_chunking_to_forward(
            #     self.feed_forward_chunk_nl, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
            # )
        elif modality_type == "code":
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk_code, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
            )
        else:
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk_tc, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
            )
        outputs = (layer_output,) + outputs

        outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

    # def feed_forward_chunk_nl(self, attention_output):
    #     intermediate_output = self.intermediate_nl(attention_output)
    #     layer_output = self.output_nl(intermediate_output, attention_output)
    #     return layer_output
    
    def feed_forward_chunk_code(self, attention_output):
        intermediate_output = self.intermediate_code(attention_output)
        layer_output = self.output_code(intermediate_output, attention_output)
        return layer_output
    
    def feed_forward_chunk_tc(self, attention_output):
        intermediate_output = self.intermediate_cn(attention_output)
        layer_output = self.output_cn(intermediate_output, attention_output)
        return layer_output

class NewRobertaAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        self.self = NewRobertaSelfAttention(config, is_cross_attention)
        self.output = RobertaSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class NewRobertaSelfAttention(nn.Module):
    def __init__(self, config, is_cross_attention):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        # self.save_attention = False

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        outputs = outputs + (past_key_value,)
        return outputs
