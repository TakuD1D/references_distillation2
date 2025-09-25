import peft
def cnn_preprocess_data(data, tokenizer):
    """
    Preprocess the data for T5 model.
    Args:
        data (DataFrame): Data containing 'article' and 'highlights'.
        tokenizer (T5Tokenizer): Tokenizer for T5 model.
    Returns:
        dict: Tokenized inputs and labels.
    """
    input_data = ["summarize: " + document for document in data["article"]]
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

def xsum_preprocess_data(data, tokenizer):
    """
    Preprocess the data for T5 model.
    Args:
        data (DataFrame): Data containing 'article' and 'highlights'.
        tokenizer (T5Tokenizer): Tokenizer for T5 model.
    Returns:
        dict: Tokenized inputs and labels.
    """
    input_data = ["summarize: " + document for document in data["document"]]
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



def freeze_layer(model):
    for name, param in model.named_parameters():
        if "shared" in name or "lm_head" in name:
            param.requires_grad = False
            print(f"freeze {name}")
    return model

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
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="all", # "lora_only" or "all"
        task_type=peft.TaskType.SEQ_2_SEQ_LM,
    )
    lora_model = peft.get_peft_model(model, lora_config)
    lora_model.print_trainable_parameters()
    return lora_model