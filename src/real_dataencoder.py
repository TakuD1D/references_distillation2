from transformers import T5Model, T5Tokenizer, AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, BartModel, T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqModelOutput, Seq2SeqLMOutput
import torch
from torch.utils.data import DataLoader
import numpy as np 
import datasets
import peft

from tqdm import tqdm


def get_lora_model(model):
    """
    Get a LoRA model by freezing certain parameters.
    Args:
        model (T5Model): The T5 model to modify.
    Returns:
        T5Model: The modified model with certain parameters frozen.
    """
    for name, param in model.named_parameters():
        param.requires_grad = False
    lora_config =peft.LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type=peft.TaskType.SEQ_2_SEQ_LM,
    )
    lora_model = peft.get_peft_model(model, lora_config)
    lora_model.print_trainable_parameters()
    return lora_model

def preprocess_input_data(
    data,
    tokenizer,
):
    # xsum: document, summary
    # cnn_dailymail: article, highlights
    inputs = dict()
    inputs = tokenizer(
        data["document"] if "document" in data else data["article"],
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    inputs["labels"] = tokenizer(
        data["summary"] if "summary" in data else data["highlights"],
        max_length=64,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )["input_ids"]
    
    return inputs

def shift_tokens_right(input_ids, pad_token_id, decoder_start_token_id):
    shifted = input_ids.new_zeros(input_ids.shape)
    shifted[:, 1:] = input_ids[:, :-1]
    shifted[:, 0] = decoder_start_token_id if decoder_start_token_id is not None else pad_token_id
    return shifted

def preprocess_vectorinput_data(
    dataloader,
    embedding_layer,
    model
):
    inputs = dict()
    
    inputs_embeds = []
    decoder_inputs_embeds = []
    labels = []
    for batch in tqdm(dataloader, desc="Processing batches"):
        batch = {k: v.to("cuda") for k, v in batch.items()}
        with torch.no_grad():
            inputs_embeds.append(embedding_layer(
                batch["input_ids"],
            ).cpu())
            shift_labels = shift_tokens_right(batch["labels"], 
                                            pad_token_id=model.config.pad_token_id, 
                                            decoder_start_token_id=model.config.decoder_start_token_id)
            decoder_inputs_embeds.append(embedding_layer(
                torch.tensor(shift_labels).to("cuda"),
                # return_tensors="pt",
            ).cpu())
            batch.pop("labels")  # Remove labels from batch to avoid passing them to the model
            
            labels.append(model(
                input_ids=batch["input_ids"],
                decoder_input_ids=shift_labels,
            ).last_hidden_state.cpu())
    inputs["inputs_embeds"] = np.concatenate(inputs_embeds, axis=0)
    inputs["decoder_inputs_embeds"] = np.concatenate(decoder_inputs_embeds, axis=0)
    inputs["labels"] = np.concatenate(labels, axis=0)
    return inputs



class VecContrasT5ModelLM(T5ForConditionalGeneration):
    def forward(self, **kwargs):
        if "inputs_embeds" in kwargs and kwargs.get("decoder_inputs_embeds", None) is not None:
            outputs = self.vec_forward(**kwargs)
            return outputs.loss
        else:
            return self.encoder_forward(**kwargs)
        
    def encoder_forward(self, **kwargs):
        encoder_output = self.encoder(
            input_ids=kwargs.get("input_ids", None),
            attention_mask=kwargs.get("attention_mask", None),
            inputs_embeds=kwargs.get("inputs_embeds", None),
        )
        decoder_outputs = self.decoder(
            input_ids=kwargs.get("decoder_input_ids", None),
            attention_mask=kwargs.get("decoder_attention_mask", None),
            inputs_embeds=kwargs.get("decoder_inputs_embeds", None),
            encoder_hidden_states=encoder_output.last_hidden_state,
            encoder_attention_mask=kwargs.get("encoder_attention_mask", None),
        )
        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_output.last_hidden_state,
            encoder_hidden_states=encoder_output.hidden_states,
            encoder_attentions=encoder_output.attentions,
        )
        
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

if __name__ == "__main__":
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
    # tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    model = T5Model.from_pretrained("google-t5/t5-base").to("cuda")
    # model = BartModel.from_pretrained("facebook/bart-base").to("cuda")
    # model = VecContrasT5ModelLM.from_pretrained("google-t5/t5-base").to("cuda")
    # model = get_lora_model(model)
    syn_size = 50
    embedding_layer = model.get_input_embeddings()
    collate = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
    )
    for i in range(1,5):

        # Load the dataset
        dataset = datasets.load_dataset("xsum", split="train")
        # dataset = datasets.load_dataset("abisee/cnn_dailymail", "3.0.0", split="train")
        dataset = dataset.shuffle()  # Shuffle the dataset for randomness
        dataset = dataset.select(range(syn_size))  # Select a subset for testing
        

        # Preprocess the data
        inputs = dataset.map(
            lambda x: preprocess_input_data(x, tokenizer),
            batched=True,
            # batch_size=8,
            remove_columns=dataset.column_names,
        )
        loader = DataLoader(
            inputs,
            batch_size=2,
            shuffle=False,
            collate_fn=collate,
        )
        
        
        vector = preprocess_vectorinput_data(
            loader,
            embedding_layer,
            model
        )
        
        torch.save(vector, f"../data/base_embed/xsum/t5base/test_xsum_t5base_{syn_size}_embed_base_ver{i}.pt")
