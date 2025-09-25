from transformers import T5Tokenizer, AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq
import torch
from torch.utils.data import DataLoader
import numpy as np 
import datasets
from datasets import load_dataset
import evaluate
import peft
import os
import wandb
# from modeling.vec_t5 import VecT5ModelLM as VecT5Model
from modeling.vec_t5 import VecContrasMeanT5ModelLM as VecT5Model
from timm.scheduler import CosineLRScheduler

from tqdm import tqdm


def preprcess_ininput_data(
    data,
    tokenizer
):
     # xsum: document, summary
     #cnn_dailymail: article, highlights
    input_data = [ "summarize: "+ document for document in data["document"]]
    inputs = tokenizer(
        input_data,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    inputs["labels"] = tokenizer(
        data["summary"],
        max_length=64,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )["input_ids"]
    return inputs 

def cnn_preprocess_input_data(
    data,
    tokenizer
):
    # cnn_dailymail: article, highlights
    input_data = [ "summarize: "+ article for article in data["article"]]
    inputs = tokenizer(
        input_data,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    inputs["labels"] = tokenizer(
        data["highlights"],
        max_length=64,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )["input_ids"]
    return inputs






def synthetic_data(embed, n):
    random_batch = torch.randperm(len(embed))
    batch_list = list(torch.split(random_batch, n))
    return batch_list

def get_lora_model(model):
    # lora付けたmodel取得
    for name, param in model.named_parameters():
        param.requires_grad = False
    config = peft.LoraConfig(
        task_type="SEQ_2_SEQ_LM",
        r=8,
        lora_alpha=32,
        # target_modules=["q", "v"],
        lora_dropout=0.1,
    )
    lora_model = peft.get_peft_model(
        model, 
        config
    )
    return lora_model

def freeze_layer(model):
    for name, param in model.named_parameters():
        if "shared" in name or "lm_head" in name:
            param.requires_grad = False
            print(f"freeze {name}")
        # else:
        #     param.requires_grad = False
    return model

def hybrid_freeze_layer(model):
    for name, param in model.named_parameters():
        if "shared" in name or "lm_head" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            print(f"freeze {name}")
    return model
def hybrid_encoder_freeze_layer(model):
    for name, params in model.named_parameters():
        # decoder + encoderのfreeze
        if "shared" in name or "lm_head" in name:
            params.requires_grad = True
        elif "encoder" in name:
            params.requires_grad = False
            print(f"freeze {name}")
            
    return model




def train(lr=1e-5, epoch=3, batch_size=2, model_name="google-t5/t5-base", dataset_name="xsum", embed_data_path=None, syn_size=None, output_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model setup
    embed_data = torch.load(embed_data_path, weights_only=False)
    model = VecT5Model.from_pretrained(model_name)
    # model = freeze_layer(model)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = get_lora_model(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    
    
    model.train()
    model.to(device)
    
    
    # データセット
    if "xsum" in dataset_name:
        dataset = datasets.load_dataset(dataset_name, trust_remote_code=True)["validation"]
        
        # eval dataset
        dataset = dataset.shuffle().select(range(200))  # 200 samples for eval
        dataset = dataset.map(
            lambda x: preprcess_ininput_data(x, tokenizer),
            batched=True,
            batch_size=8,
            remove_columns=dataset.column_names,
        )
    else:
        dataset = datasets.load_dataset(dataset_name, "3.0.0")["validation"]
        dataset = dataset.shuffle().select(range(200))  # 200 samples for eval
        dataset = dataset.map(
            lambda x: cnn_preprocess_input_data(x, tokenizer),
            batched=True,
            batch_size=8,
            remove_columns=dataset.column_names,
        )
    eval_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=DataCollatorForSeq2Seq(tokenizer, model),
    )
    # batch_list = synthetic_data(embed_data["inputs_embeds"], batch_size)
    # scheduler
    # scheduler = CosineLRScheduler(optimizer, t_initial=epoch*len(batch_list), lr_min=1e-6, warmup_lr_init=1e-6, warmup_t=10)
    best_loss = float('inf')

    for ep in tqdm(range(epoch),desc="Epochs"):
        
        # current_lr = get_scheduled_lr(ep, initial_lr=lr)
        batch_list = synthetic_data(embed_data["inputs_embeds"], batch_size)
        for step, batchs in enumerate(batch_list):
            data = { 
                "inputs_embeds": embed_data["inputs_embeds"][batchs],
                "decoder_inputs_embeds": embed_data["decoder_inputs_embeds"][batchs],
                "labels": embed_data["labels"][batchs]
            }
            # print(data["inputs_embeds"].shape, data["decoder_inputs_embeds"].shape, data["labels"].shape)
            batch = {k: v.detach().clone().to(device) if isinstance(v, torch.Tensor) else torch.tensor(v).to(device) for k, v in data.items() if v is not None}
            output = model(**batch)
            print(f"Output: {output}")
            wandb.log({"Constractive Loss": output})
            loss = output 
            # model.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step(ep*len(batch_list) + step)
            # optimizer.param_groups[0]['lr'] = scheduler.get_lr()  # 学習率を更新
            # wandb.log({"syn/lr": optimizer.param_groups[0]['lr']})

        # evaluation
        if (ep % 1 == 0 and ep != 0) or (ep == epoch-1):
            avg_mean = []
            for eval_batch in tqdm(eval_loader, desc="Eval"):
                eval_batch = {k: v.to(device) for k, v in eval_batch.items() if v is not None}
                with torch.no_grad():
                    loss = model(**eval_batch).loss
                avg_mean.append(loss.item())
            avg_mean = np.mean(avg_mean)
            wandb.log({"eval/syn/CE": avg_mean})
            if avg_mean < best_loss:
                best_loss = avg_mean
                model.save_pretrained(f"{output_path}/best")
                tokenizer.save_pretrained(f"{output_path}/best")
                print(f"Best model saved with loss: {best_loss}")

    model = VecT5Model.from_pretrained(f"{output_path}/best")
    return model
    
def real_train(CL_model, lr=1e-5, epoch=3, batch_size=2, model_name="google-t5/t5-base", dataset_name="xsum", real_size=None, output_path=None, summary_eval=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model setup
    model = CL_model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = hybrid_encoder_freeze_layer(model)
    model = get_lora_model(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    model.to(device)
    # データセット
    if "xsum" in dataset_name:
        dataset = datasets.load_dataset(dataset_name, trust_remote_code=True)["validation"]
        train_dataset = datasets.load_dataset(dataset_name, trust_remote_code=True)["train"]

        # eval dataset
        dataset = dataset.shuffle().select(range(200))  # 200 samples for eval
        train_dataset = train_dataset.shuffle().select(range(real_size))  # real_size samples for train
        dataset = dataset.map(
            lambda x: preprcess_ininput_data(x, tokenizer),
            batched=True,
            batch_size=8,
            remove_columns=dataset.column_names,
        )
        train_dataset = train_dataset.map(
            lambda x: preprcess_ininput_data(x, tokenizer),
            batched=True,
            batch_size=8,
            remove_columns=train_dataset.column_names,
        )
    else:
        dataset = datasets.load_dataset(dataset_name, "3.0.0")["validation"]
        train_dataset = datasets.load_dataset(dataset_name, "3.0.0")["train"]
        dataset = dataset.shuffle().select(range(200))  # 200 samples for eval
        train_dataset = train_dataset.shuffle().select(range(real_size))  # real_size samples for train
        dataset = dataset.map(
            lambda x: cnn_preprocess_input_data(x, tokenizer),
            batched=True,
            batch_size=8,
            remove_columns=dataset.column_names,
        )
        train_dataset = train_dataset.map(
            lambda x: cnn_preprocess_input_data(x, tokenizer),
            batched=True,
            batch_size=8,
            remove_columns=train_dataset.column_names,
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=DataCollatorForSeq2Seq(tokenizer, model),
    )
    eval_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=DataCollatorForSeq2Seq(tokenizer, model),
    )
    # scheduler
    scheduler = CosineLRScheduler(optimizer, t_initial=epoch*len(train_loader), lr_min=1e-6, warmup_lr_init=1e-6, warmup_t=10)
    
    best_loss = float('inf')

    for ep in tqdm(range(epoch), desc="Epochs"):
        for step, train_batch in enumerate(train_loader):
            train_batch = {k: v.to(device) for k, v in train_batch.items() if v is not None}
            # with torch.no_grad():
            output = model(**train_batch)
            loss = output.loss
            # model.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(ep*len(train_loader)+step)
            # optimizer.param_groups[0]['lr'] = scheduler.get_lr()  # 学習率を更新
            wandb.log({"train_loss/CE": loss.item()})
            wandb.log({"real/lr": optimizer.param_groups[0]['lr']})
            
            
        if (ep % 5 == 0 and ep != 0) or (ep == epoch-1):
            avg_mean = []
            for eval_batch in tqdm(eval_loader, desc="Eval"):
                eval_batch = {k: v.to(device) for k, v in eval_batch.items() if v is not None}
                with torch.no_grad():
                    loss = model(**eval_batch).loss
                avg_mean.append(loss.item())
            avg_mean = np.mean(avg_mean)
            wandb.log({"eval/real/CE": avg_mean})
            # early stopping
            if avg_mean < best_loss:
                best_loss = avg_mean
                model.save_pretrained(f"{output_path}/best")
                tokenizer.save_pretrained(f"{output_path}/best")
                print(f"Best model saved with loss: {best_loss}")
            
            
            
    wandb.finish()
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    
    
def init_wandb(model_name, dataset_name, size, epoch, lr, syn_dataset, real_size=None, real_epoch=None, real_lr=None,real_batch=None):
    wandb.init(
        project="2stagetrain_ver1",
        name=f"LoRA_{model_name}_{dataset_name}_synsize{size}_real{real_size}_synepoch{epoch}_lr{lr}_realepoch{real_epoch}_reallr{real_lr}_temp0.1",
        group=f"syn_50_test_temp0.1",
        config={
            "model_name": model_name,
            "dataset_name": dataset_name,
            "data_per_label:datasize": size,
            "syn_epoch": epoch,
            "real_epoch": real_epoch,
            "lr": lr,
            "real_lr": real_lr,
            "data":syn_dataset,
            "real_size": real_size,
            "cl_batch_size": 4,
            "real_batch_size": real_batch,
            "output":False,
            "schedule":"CosineLR",
            "early_stop":True,
        },
        reinit=True
    )
    
if __name__ == "__main__":
    lrs = [1e-4] # synのハイパラ
    epochs = [100] # syn50の入パラ
    syn_sizes = [50] # xsumのsyn_size  50,100,200
    syn_epochs = [100] # syn_sizeごとにepochを変えているのでしているする
    real_size = [20, 50,100,200,800]
    real_epochs = [100] #3,5,10
    real_lrs = [1e-4, 3e-3, 5e-4, 1e-3, 1e-5,5e-5]
    model_name = "google-t5/t5-base"
    # model_name = "facebook/bart-base"
    
    # dataset_name = "abisee/cnn_dailymail" # xsum abisee/cnn_dailymail EdinburghNLP/xsum
    dataset_name = "xsum" # xsum abisee/cnn_dailymail EdinburghNLP/xsum
    
    for index, (syn_size, syn_epoch) in enumerate(zip(syn_sizes,syn_epochs)):
        # embed_data_path = f"../data/distil_data/xsum/t5base/syn{syn_size}_epoch{syn_epoch}_std0.pt"
        # embed_data_path = f"../data/embed_data/cnn/test_cnn_t5base_{syn_size}_embed_base_ver0.pt"
        # embed_data_path = f"../data/distil_data/xsum/t5base/best/Mean_syn{syn_size}_epoch{syn_epoch}_std0_ver2_best.pt"
        # embed_data_path = f"/workspace/distil_train/src/prop_method/data/distil_data/xsum/t5base/overfitting/syn{syn_size}_epoch_{syn_epoch}.pt"
        embed_data_path = f"../data/distil_data/xsum/t5base/best/Mean_syn{syn_size}_epoch{syn_epoch}_std0_ver2_best.pt"
        for lr in lrs:
            # lr = lrs[index]
            epoch = epochs[index]
            for real_data in real_size:
                for real_epoch in real_epochs:
                    for real_lr in real_lrs:
                        print(f"Training with lr: {lr}, epoch: {epoch}")
                        output_path = f"../model/proposed/t5base/xsum/add/LoRA_syn{syn_size}-real{real_data}-syn_epoch{epoch}-lr{lr}-real_epoch{real_epoch}_reallr{real_lr}" # normal distil = /best/
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        init_wandb(model_name=model_name, dataset_name=dataset_name, size=syn_size, epoch=epoch, lr=lr, syn_dataset=embed_data_path, real_size=real_data, real_epoch=real_epoch, real_lr=real_lr, real_batch=4)
                        model = train(lr=lr, epoch=epoch, batch_size=4, model_name=model_name, dataset_name=dataset_name, embed_data_path=embed_data_path, syn_size=syn_size, output_path=output_path)
                        # real_train用のモデルのepochの入パラ
                        real_train(
                            CL_model=model,
                            lr=real_lr, 
                            epoch=real_epoch, 
                            batch_size=4, 
                            model_name=model_name, 
                            dataset_name=dataset_name, 
                            real_size=real_data, 
                            output_path=output_path
                        )
                        
                        
                        print(f"Finished training with lr: {lr}, epoch: {epoch}")