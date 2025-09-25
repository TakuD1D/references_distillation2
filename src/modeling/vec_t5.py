
from transformers import T5Model, T5Tokenizer, AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput, Seq2SeqModelOutput
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np 
import datasets
import peft
import os
import wandb


from tqdm import tqdm


class VecT5ModelLM(T5Model):
    def forward(self, **kwargs):
        if "lm_haed" in kwargs:
            output = super().forward(**kwargs)
            last_hidden_state = output.last_hidden_state
            
            output.loss = loss
            return output
        else:
            if "inputs_embeds" in kwargs and "decoder_inputs_embeds" in kwargs:
                output = super().forward(inputs_embeds=kwargs["inputs_embeds"], decoder_inputs_embeds=kwargs["decoder_inputs_embeds"])
                last_hidden_state = output.last_hidden_state
                loss = self.compute_loss(last_hidden_state, kwargs["labels"])
                output.loss = loss
                return output
            else:
                return super().forward(**kwargs)
    def compute_loss(self, last_hidden_state, labels):
        # Compute the loss using the last hidden state and labels
        cos = torch.nn.CosineSimilarity(dim=1)
        loss =  1 - cos(last_hidden_state, labels).mean()
        return loss
    
    
class VecT5ModelLMGene(T5ForConditionalGeneration):
    def vec_forward(self, **kwargs):
        encoder_outputs = self.encoder(
            inputs_embeds=kwargs["inputs_embeds"],
            # decoder_inputs_embeds=kwargs["decoder_inputs_embeds"]
        )
        decoder_outputs = self.decoder(
            input_ids=kwargs.get("input_ids", None),
            attention_mask=kwargs.get("attention_mask", None),
            inputs_embeds=kwargs.get("decoder_inputs_embeds", None),
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=kwargs.get("encoder_attention_mask", None),
        )
        if "labels" in kwargs:
            loss = self.vec_compute_loss(decoder_outputs.last_hidden_state, kwargs["labels"])
            return Seq2SeqLMOutput(
            loss=loss,
            # logits=None,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            )
        
        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
        
    def vec_compute_loss(self, last_hidden_state, labels):
        # Compute the loss using the last hidden state and labels
        b,t,h = last_hidden_state.shape
        last_hidden_state = last_hidden_state.view(-1, h)
        labels = labels.view(-1,h)
        cos = torch.nn.CosineSimilarity(dim=1)
        loss =  (1 - cos(last_hidden_state, labels)).mean()
        # 正則化項
        mse_loss = ((last_hidden_state - labels) ** 2).mean()
        
        
        return 10.0 * loss + 0.5 * mse_loss
    
    
class VecT5ModelLM(T5ForConditionalGeneration):
    def forward(self, **kwargs):
        if "inputs_embeds" in kwargs and kwargs.get("decoder_inputs_embeds", None) is not None:
            outputs = self.vec_forward(**kwargs)
            return outputs.loss
        else:
            return super().forward(**kwargs)
        
    def vec_forward(self, **kwargs):
        encoder_outputs = self.encoder(
            inputs_embeds=kwargs["inputs_embeds"],
            # decoder_inputs_embeds=kwargs["decoder_inputs_embeds"]
        )
        decoder_outputs = self.decoder(
            input_ids=kwargs.get("input_ids", None),
            attention_mask=kwargs.get("attention_mask", None),
            inputs_embeds=kwargs.get("decoder_inputs_embeds", None),
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=kwargs.get("encoder_attention_mask", None),
        )
        if "labels" in kwargs:
            loss = self.vec_compute_loss(decoder_outputs.last_hidden_state, kwargs["labels"])
            return Seq2SeqLMOutput(
            loss=loss,
            # logits=None,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            )
        
        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
        
    def vec_compute_loss(self, last_hidden_state, labels):
        # Compute the loss using the last hidden state and labels
        b,t,h = last_hidden_state.shape
        last_hidden_state = last_hidden_state.view(-1, h)
        labels = labels.view(-1,h)
        cos = torch.nn.CosineSimilarity(dim=1)
        loss =  (1 - cos(last_hidden_state, labels)).mean()
        # 正則化項
        # mse_loss = ((last_hidden_state - labels) ** 2).mean()
        
        
        return loss
    
    
class VecsubmeanLossT5ModelLM(T5ForConditionalGeneration):
    def forward(self, **kwargs):
        if "inputs_embeds" in kwargs and kwargs.get("decoder_inputs_embeds", None) is not None:
            outputs = self.vec_forward(**kwargs)
            return outputs.loss
        else:
            return super().forward(**kwargs)
        
    def vec_forward(self, **kwargs):
        encoder_outputs = self.encoder(
            inputs_embeds=kwargs["inputs_embeds"],
            # decoder_inputs_embeds=kwargs["decoder_inputs_embeds"]
        )
        decoder_outputs = self.decoder(
            input_ids=kwargs.get("input_ids", None),
            attention_mask=kwargs.get("attention_mask", None),
            inputs_embeds=kwargs.get("decoder_inputs_embeds", None),
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=kwargs.get("encoder_attention_mask", None),
        )
        if "labels" in kwargs:
            loss = self.vec_compute_loss(encoder_last_hidden_state=encoder_outputs, last_hidden_state=decoder_outputs.last_hidden_state, labels=kwargs["labels"])
            return Seq2SeqLMOutput(
            loss=loss,
            # logits=None,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            )
        
        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
        
    def vec_compute_loss(self, encoder_last_hidden_state, last_hidden_state, labels):
        # Σ <label token + encoder head , next Token token vector>
        b,t,h = last_hidden_state.shape
        last_hidden_state = last_hidden_state.view(-1, h)
        labels = labels.view(-1,h)
        encoder_last_hidden_state = encoder_last_hidden_state.last_hidden_state[:,0,:]
        encoder_last_hidden_state = encoder_last_hidden_state.repeat_interleave(t, dim=0) # b*t, h
        cos = torch.nn.CosineSimilarity(dim=1)
        prop_loss = (1-cos(encoder_last_hidden_state + labels, last_hidden_state)).mean()
        return prop_loss
    
    
class VecsubsumLossT5ModelLM(T5ForConditionalGeneration):
    def forward(self, **kwargs):
        if "inputs_embeds" in kwargs and kwargs.get("decoder_inputs_embeds", None) is not None:
            outputs = self.vec_forward(**kwargs)
            return outputs.loss
        else:
            return super().forward(**kwargs)
        
    def vec_forward(self, **kwargs):
        encoder_outputs = self.encoder(
            inputs_embeds=kwargs["inputs_embeds"],
            # decoder_inputs_embeds=kwargs["decoder_inputs_embeds"]
        )
        decoder_outputs = self.decoder(
            input_ids=kwargs.get("input_ids", None),
            attention_mask=kwargs.get("attention_mask", None),
            inputs_embeds=kwargs.get("decoder_inputs_embeds", None),
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=kwargs.get("encoder_attention_mask", None),
        )
        if "labels" in kwargs:
            loss = self.vec_compute_loss(encoder_last_hidden_state=encoder_outputs, last_hidden_state=decoder_outputs.last_hidden_state, labels=kwargs["labels"])
            return Seq2SeqLMOutput(
            loss=loss,
            # logits=None,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            )
        
        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
        
    def vec_compute_loss(self, encoder_last_hidden_state, last_hidden_state, labels):
        # Σ <label token + encoder head , next Token token vector>
        b,t,h = last_hidden_state.shape
        last_hidden_state = last_hidden_state.view(-1, h)
        labels = labels.view(-1,h)
        encoder_last_hidden_state = encoder_last_hidden_state.last_hidden_state[:,0,:]
        encoder_last_hidden_state = encoder_last_hidden_state.repeat_interleave(t, dim=0) # b*t, h
        cos = torch.nn.CosineSimilarity(dim=1)
        prop_loss = torch.sum(1-cos(encoder_last_hidden_state + labels, last_hidden_state))
        return prop_loss
    
    
    
class VecContrasT5ModelLM(T5ForConditionalGeneration):
    def forward(self, **kwargs):
        if "inputs_embeds" in kwargs and kwargs.get("decoder_inputs_embeds", None) is not None:
            outputs = self.vec_forward(**kwargs)
            return outputs.loss
        else:
            return super().forward(**kwargs)
        
    def vec_forward(self, **kwargs):
        encoder_outputs = self.encoder(
            inputs_embeds=kwargs["inputs_embeds"],
            # decoder_inputs_embeds=kwargs["decoder_inputs_embeds"]
        )
        decoder_outputs = self.decoder(
            input_ids=kwargs.get("input_ids", None),
            attention_mask=kwargs.get("attention_mask", None),
            inputs_embeds=kwargs.get("decoder_inputs_embeds", None),
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=kwargs.get("encoder_attention_mask", None),
        )
        if "labels" in kwargs:
            loss = self.vec_compute_loss(decoder_last_hidden_state=decoder_outputs.last_hidden_state, labels=kwargs["labels"])
            return Seq2SeqLMOutput(
            loss=loss,
            # logits=None,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            )
        
        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
        
    def vec_compute_loss(self, decoder_last_hidden_state,labels, temp=0.07):
        # contrastive loss
        
        b,t,h = decoder_last_hidden_state.shape
        querys = F.normalize(decoder_last_hidden_state.view(-1, h), dim=1)
        keys = F.normalize(labels.view(-1, h), dim=1)
        
        sim_matrix = torch.matmul(querys, keys.T)  # b*t, b*t
        targets = torch.arange(b*t, device=sim_matrix.device)
        
        loss = F.cross_entropy(sim_matrix/temp, targets)
        return loss
    
class VecContrasEncT5ModelLM(T5ForConditionalGeneration):
    def forward(self, **kwargs):
        if "inputs_embeds" in kwargs and kwargs.get("decoder_inputs_embeds", None) is not None:
            outputs = self.vec_forward(**kwargs)
            return outputs.loss
        else:
            return super().forward(**kwargs)
        
    def vec_forward(self, **kwargs):
        encoder_outputs = self.encoder(
            inputs_embeds=kwargs["inputs_embeds"],
            # decoder_inputs_embeds=kwargs["decoder_inputs_embeds"]
        )
        decoder_outputs = self.decoder(
            input_ids=kwargs.get("input_ids", None),
            attention_mask=kwargs.get("attention_mask", None),
            inputs_embeds=kwargs.get("decoder_inputs_embeds", None),
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=kwargs.get("encoder_attention_mask", None),
        )
        if "labels" in kwargs:
            loss = self.vec_compute_loss(encoder_last_hidden_state=encoder_outputs,decoder_last_hidden_state=decoder_outputs.last_hidden_state, labels=kwargs["labels"])
            return Seq2SeqLMOutput(
            loss=loss,
            # logits=None,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            )
        
        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
        
    def vec_compute_loss(self,encoder_last_hidden_state, decoder_last_hidden_state,labels, temp=0.07):
        # contrastive loss
        
        b,t,h = decoder_last_hidden_state.shape
        querys = F.normalize(decoder_last_hidden_state.view(-1, h), dim=1)
        keys = F.normalize(labels.view(-1, h), dim=1)
        encoder_last_hidden_state = encoder_last_hidden_state.last_hidden_state[:,0,:]
        encoder_last_hidden_state = encoder_last_hidden_state.repeat_interleave(t, dim=0) # b*t, h
        querys = querys + encoder_last_hidden_state
        sim_matrix = torch.matmul(querys, keys.T)  # b*t, b*t
        targets = torch.arange(b*t, device=sim_matrix.device)
        
        loss = F.cross_entropy(sim_matrix/temp, targets)
        return loss
        
        
class VecContrasMeanT5ModelLM(T5ForConditionalGeneration):
    def forward(self, **kwargs):
        if "inputs_embeds" in kwargs and kwargs.get("decoder_inputs_embeds", None) is not None:
            outputs = self.vec_forward(**kwargs)
            return outputs.loss
        else:
            return super().forward(**kwargs)
        
    def vec_forward(self, **kwargs):
        encoder_outputs = self.encoder(
            inputs_embeds=kwargs["inputs_embeds"],
            # decoder_inputs_embeds=kwargs["decoder_inputs_embeds"]
        )
        decoder_outputs = self.decoder(
            input_ids=kwargs.get("input_ids", None),
            attention_mask=kwargs.get("attention_mask", None),
            inputs_embeds=kwargs.get("decoder_inputs_embeds", None),
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=kwargs.get("encoder_attention_mask", None),
        )
        if "labels" in kwargs:
            loss = self.vec_compute_loss(decoder_last_hidden_state=decoder_outputs.last_hidden_state, labels=kwargs["labels"])
            return Seq2SeqLMOutput(
            loss=loss,
            # logits=None,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            )
        
        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
        
    def vec_compute_loss(self, decoder_last_hidden_state,labels, temp=0.1):
        b,t,h = decoder_last_hidden_state.shape
        querys = F.normalize(decoder_last_hidden_state.mean(dim=1), dim=1)  # (b, h)
        keys   = F.normalize(labels.mean(dim=1), dim=1)                     # (b, h)

        sim_matrix = torch.matmul(querys, keys.T)  # shape: (b, b)
        targets = torch.arange(b, device=sim_matrix.device)

        loss = F.cross_entropy(sim_matrix / temp, targets)
        return loss