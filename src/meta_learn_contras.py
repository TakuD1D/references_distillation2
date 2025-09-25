import torch
from torch.utils.data import DataLoader, Dataset
from torch.cuda import amp
import numpy as np
import datasets
import wandb
from tqdm import tqdm
import os

from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq

# file imports
from modeling.vec_t5 import VecContrasMeanT5ModelLM as VecT5ModelLM
from utils.preprocess import cnn_preprocess_data, xsum_preprocess_data, freeze_layer, get_lora_model
from utils.batch import batch_synthetic_index, batch_synthetic_index_all

def cache_cuda():
    print("cuda memory allocated: ", torch.cuda.memory_allocated())

class Trainer:
    def __init__(self, model_name, dataset_name, distil_epochs, inner_loop, data_lr=1e-3, model_lr=1e-5, init="random", syn_size=200, embed_data=None, lora=False, decoder_train=True, output=None, dataset_map=None):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.distil_epochs = distil_epochs
        self.inner_loop = inner_loop
        self.data_lr = data_lr
        self.model_lr = model_lr
        self.init = init
        self.syn_size = syn_size
        self.embed_data = embed_data
        self.output = output
        self.dataset_map = dataset_map
        # self.syn_lr = torch.tensor(1e-1, device="cuda", requires_grad=True)
        
        self.best_loss = float("inf")
        self.best_epoch = 0
        # model and tokenizer setup
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = VecT5ModelLM.from_pretrained(model_name)
        self.model = freeze_layer(self.model)
        if lora:
            self.model = get_lora_model(self.model)
        
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model.to(self.device)
        
        # real dataset setup
        if "xsum" in dataset_name and dataset_map is None:
            self.dataset = datasets.load_dataset(dataset_name,trust_remote_code=True)["train"]
            self.dataset_map = self.dataset.map(
                lambda x: xsum_preprocess_data(x, self.tokenizer),
                batched=True,
                remove_columns=self.dataset.column_names,
            )
            self.train_dataloader = DataLoader(
                self.dataset_map,
                batch_size=4,
                collate_fn=self.data_collator,
                shuffle=True,
            )
        elif "cnn_dailymail" in dataset_name and dataset_map is None:
            self.dataset = datasets.load_dataset(dataset_name, "3.0.0")["train"]
            self.dataset_map = self.dataset.map(
                lambda x: cnn_preprocess_data(x, self.tokenizer),
                batched=True,
                remove_columns=self.dataset.column_names,
            )
            self.train_dataloader = DataLoader(
                self.dataset_map,
                batch_size=4,
                collate_fn=self.data_collator,
                shuffle=True,
            )
        else:
            self.train_dataloader = DataLoader(
                self.dataset_map,
                batch_size=4,
                collate_fn=self.data_collator,
                shuffle=True,
            )
        
        # syn data setup
        if init == "random":
            self.syn_inputs_embeds = torch.nn.Parameter(torch.randn((syn_size, 512, 768), device=self.device, requires_grad=True))
            self.syn_decoder_inputs_embeds = torch.nn.Parameter(torch.randn((syn_size, 64, 768), device=self.device, requires_grad=True))
            self.syn_labels = torch.nn.Parameter(torch.randn((syn_size, 64, 768), device=self.device, requires_grad=True))
        elif init == "sample":
            embedding_data = torch.load(embed_data, weights_only=False)
            self.syn_inputs_embeds = torch.nn.Parameter(torch.tensor(embedding_data["inputs_embeds"]).to(self.device).requires_grad_(True))
            self.syn_decoder_inputs_embeds = torch.nn.Parameter(torch.tensor(embedding_data["decoder_inputs_embeds"]).to(self.device).requires_grad_(True))
            self.syn_labels = torch.nn.Parameter(torch.tensor(embedding_data["labels"]).to(self.device).requires_grad_(True))
        elif init =="sample_noise":
            embedding_data = torch.load(embed_data, weights_only=False)
            noise_std = 0.01
            self.syn_inputs_embeds = torch.tensor(embedding_data["inputs_embeds"]).to(self.device)
            self.syn_decoder_inputs_embeds = torch.tensor(embedding_data["decoder_inputs_embeds"]).to(self.device)
            self.syn_labels = torch.tensor(embedding_data["labels"]).to(self.device)
            
            # Add noise to the synthetic embeddings
            self.syn_inputs_embeds += torch.randn_like(self.syn_inputs_embeds) * noise_std
            self.syn_decoder_inputs_embeds += torch.randn_like(self.syn_decoder_inputs_embeds) * noise_std
            self.syn_labels += torch.randn_like(self.syn_labels) * noise_std
            
            self.syn_inputs_embeds = torch.nn.Parameter(self.syn_inputs_embeds.requires_grad_(True))
            self.syn_decoder_inputs_embeds = torch.nn.Parameter(self.syn_decoder_inputs_embeds.requires_grad_(True))
            self.syn_labels = torch.nn.Parameter(self.syn_labels.requires_grad_(True))
        
        if not decoder_train:
            # If decoder is not trained, set requires_grad to False
            self.syn_decoder_inputs_embeds.requires_grad = False
            self.syn_labels.requires_grad = False
            
        # optimizers setup
        self.optimizer_data = torch.optim.AdamW(
            [self.syn_inputs_embeds, self.syn_decoder_inputs_embeds, self.syn_labels],
            lr=self.data_lr,
        )
        self.scaler = amp.GradScaler(enabled=True)
        self.wandb_()
        
    def train(self):
        batch_syn_index = []
        for i in tqdm(range(self.distil_epochs)):
            log_train_loss = 0.0
            for outer_step, batch_real in enumerate(self.train_dataloader):
                self.model.train()
                
                # init
                if outer_step == 0:
                    init_dict = self.model.state_dict()
                else:
                    self.model.load_state_dict(init_dict)
                self.model.to(self.device)
                
                params = {
                    name: param for name, param in self.model.named_parameters() if param.requires_grad
                }
                buffers = dict(self.model.named_buffers())
                
                def meta_compute_loss(params, buffers, **kwargs):
                    with amp.autocast(enabled=True, dtype=torch.bfloat16):
                        # print("labels shape: ", kwargs["labels"].shape)
                        loss = torch.func.functional_call(
                            self.model,
                            (params, buffers),
                            # args=input_ids,
                            kwargs=kwargs
                        )
                        if "input_ids" in kwargs:
                            loss = loss.loss
                        else:
                            wandb.log({"train/syn_loss/contras": loss.item()})
                    print(f"outer_step: {i}, cos_loss: {loss.item()}")
                    return loss
                for inner_step in range(self.inner_loop):
                    # batch 全体を満遍なく学習できるようにする
                    if not batch_syn_index:
                        batch_syn_index = batch_synthetic_index_all(self.syn_inputs_embeds, 4)
                    batch_syn = batch_syn_index.pop()
                    batch_syn = {
                        "inputs_embeds": self.syn_inputs_embeds[batch_syn],
                        "decoder_inputs_embeds": self.syn_decoder_inputs_embeds[batch_syn],
                        "labels": self.syn_labels[batch_syn],
                    }
                    grads = torch.func.grad(meta_compute_loss)(
                        params, buffers, 
                        inputs_embeds=batch_syn["inputs_embeds"],
                        decoder_inputs_embeds=batch_syn["decoder_inputs_embeds"],
                        labels=batch_syn["labels"]
                    )
                    params = {
                        name: param - self.model_lr * grads[name] for name, param in params.items()
                    }
                    print(f"inner_step: {inner_step},")
                    print(f"cuda memory allocated: {torch.cuda.memory_allocated()/ 1024 ** 2} MB")
                # print(f"cuda memory allocated: {torch.cuda.memory_allocated()/ 1024 ** 2} MB")
                # preve_emb = self.syn_inputs_embeds.detach().cpu()
                # real data loss
                batch_real = {k: v.to(self.device) for k, v in batch_real.items()}
                loss_real = meta_compute_loss(params,buffers, **batch_real)
                print(f"inner_step: {inner_step}, loss_real: {loss_real.item()}")
                wandb.log({"train/real_loss/CE": loss_real.item()})
                
                if loss_real.item() < self.best_loss:
                    self.best_loss = loss_real.item()
                    self.best_inputs_embeds = self.syn_inputs_embeds.clone().detach()
                    self.best_decoder_inputs_embeds = self.syn_decoder_inputs_embeds.clone().detach()
                    self.best_labels = self.syn_labels.clone().detach()
                    wandb.log({
                        "best_loss_updated": self.best_loss,
                    })
                    print(f"Best loss updated: {self.best_loss}")
                    self.best_epoch = i
                
                # gradient step
                self.optimizer_data.zero_grad()
                self.scaler.scale(loss_real).backward()
                
                # gradient Clipping
                self.scaler.unscale_(self.optimizer_data)
                torch.nn.utils.clip_grad_norm_(
                    [self.syn_inputs_embeds, self.syn_decoder_inputs_embeds, self.syn_labels,], 
                    max_norm=1.0
                    
                )
                
                # update parameters
                self.scaler.step(self.optimizer_data)
                self.scaler.update()
                # wandb.log({
                #     "syn_lr": self.syn_lr.item()
                # })
                # torch.cuda.empty_cache()
                # self.show_embed_prev(preve_emb)
                if outer_step > 1000:
                    # いらないかもしれない
                    self.train_dataloader = DataLoader(
                        self.dataset_map,
                        batch_size=4,   
                        collate_fn=self.data_collator,
                        shuffle=True,
                    )
                    break
            if i % 10 == 0:
                self.save_syn_embed(
                    syn_embed=self.syn_inputs_embeds,
                    decoder_syn=self.syn_decoder_inputs_embeds,
                    label_syn=self.syn_labels,
                    # syn_lr=self.syn_lr,
                    filepath=f"{self.output}/epoch{i}.pt"
                )
        self.save_syn_embed(
            syn_embed=self.syn_inputs_embeds,
            decoder_syn=self.syn_decoder_inputs_embeds,
            label_syn=self.syn_labels,
            # syn_lr=self.syn_lr,
            filepath=f"{self.output}/over.pt"
        )
        if self.best_inputs_embeds is not None:
            print(f"Best loss: {self.best_loss}, saving best model...")
            wandb.log({
                "best_loss": self.best_loss,
                "best_epoch": self.best_epoch,
                # "best_syn_lr": self.syn_lr.item()
            })
            self.save_syn_embed(
                syn_embed=self.best_inputs_embeds,
                decoder_syn=self.best_decoder_inputs_embeds,
                label_syn=self.best_labels,
                # syn_lr=self.syn_lr,
                filepath=f"{self.output}/best.pt"
            )
                    

    def save_syn_embed(self, syn_embed, decoder_syn, label_syn, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            "inputs_embeds": syn_embed.detach().cpu(),
            "decoder_inputs_embeds": decoder_syn.detach().cpu(),
            "labels": label_syn.detach().cpu(),
            # "syn_lr": syn_lr.detach().cpu()
        }, filepath)
    def filed_save_syn_embed(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            "inputs_embeds": self.syn_inputs_embeds.detach().cpu(),
            "decoder_inputs_embeds": self.syn_decoder_inputs_embeds.detach().cpu(),
            "labels": self.syn_labels.detach().cpu()
        }, filepath)
        
    def show_embed_prev(self,preve_emb):
        # prev_syn_embed, self.syn_embed: [N, ...]
        syn_embed = self.syn_inputs_embeds.detach().cpu()
        diff = (syn_embed.to("cpu") - preve_emb).abs().sum(dim=tuple(range(1, syn_embed.dim())))
        changed = diff > 1e-6  # 変化があったかどうか
        print("=== syn_embedの変化状況 ===")
        # print(f"バッチで選択されたインデックス: {batch_indices.tolist()}")
        print(f"変化したインデックス: {torch.where(changed)[0].tolist()}")
        print(f"変化していないインデックス: {torch.where(~changed)[0].tolist()}")
        print(f"変化量（非ゼロのみ）: {diff[changed].tolist()}")
        
        
        
        
        
    def wandb_(self):
        wandb.init(
            project="meta_learn_contrast",
            name=f"{self.model_name}_{self.dataset_name}-syn{self.syn_size}-",
            group=f"Mean_{self.model_name}_{self.dataset_name}_LearningRate_Update",
            config={
                "model_name": self.model_name,
                "dataset_name": self.dataset_name,
                "distil_epoch": self.distil_epochs,
                "inner_loop": self.inner_loop,
                "syn_size": self.syn_size,
                "output":self.output,
                "data_lr": self.data_lr,
                "model_lr": self.model_lr,
                "init": self.init,
                "syn_lr": True
            },
            reinit=True
        )