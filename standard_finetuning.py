import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import json
from collections import defaultdict
import argparse
# CONFIGURATION

KL_WEIGHT = 0.1
MAX_LENGTH = 512
BATCH_SIZE = 1
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
def main(args):
    FORCE_CPU = False
    if FORCE_CPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # DEVICE SETUP
    if torch.cuda.is_available() and not FORCE_CPU:
        device = torch.device("cuda")
        torch_dtype = torch.float16
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        torch_dtype = torch.float32
        print("Using CPU")
    # TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # DATASET LOADING
    train_dataset = load_dataset(args.dataset_name, split=f"train[:{args.number_of_training_samples}]")

    def tokenize_function(examples):
        if "text" in examples:
            texts = examples["text"]
        elif "prompt" in examples:
            texts = examples["prompt"]
        elif "instruction" in examples:
            texts = [
                f"{i}\n{inp if inp else ''}"
                for i, inp in zip(
                    examples["instruction"],
                    examples.get("input", [""] * len(examples["instruction"]))
                )   
            ]
        else:
            raise ValueError(f"Unknown dataset format {examples.keys()}")

        return tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    tokenized = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )

    split = tokenized.train_test_split(test_size=0.1, seed=42)
    train_dataset_tokenized = split["train"]
    val_dataset_tokenized = split["test"]

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
        )

    model_standard = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True
)

    training_args_standard = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=False,
        no_cuda=(device.type == "cpu"),
        report_to="none",
    )

    trainer_standard = Trainer(
        model=model_standard,
        args=training_args_standard,
        train_dataset=train_dataset_tokenized,
        eval_dataset=val_dataset_tokenized,
        data_collator=data_collator,
    )

    trainer_standard.train()
    trainer_standard.save_model(args.output_dir)
    torch.cuda.empty_cache()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Standard finetuning")
    parser.add_argument('--model_name', type=str, description='Pass the model name')
    parser.add_argument('--dataset_name',type=str, description='Pass the HF dataset name' )
    parser.add_argument('--number_of_training_samples', type=int, description='No of training samples')
    parser.add_argument('--output_dir', type=str, description='Path to output directory')
    args = parser.parse_args()
    main(args)