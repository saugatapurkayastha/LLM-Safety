import os
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, Dataset
from tqdm import tqdm
import json
import argparse

# =======================
# CONFIGURATION
# =======================
KL_WEIGHT = 0.1
MAX_LENGTH = 512
BATCH_SIZE = 1
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
FORCE_CPU = False
USE_FLOAT32 = True  # Set to True if experiencing NaN issues with float16
def main(args):
    if torch.cuda.is_available() and not FORCE_CPU:
        device = torch.device('cuda')
        # Use float32 if experiencing NaN issues, or if USE_FLOAT32 is True
        torch_dtype = torch.float32 if USE_FLOAT32 else torch.float16
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Using dtype: {torch_dtype} (float32 recommended for stability)")
    else:
        device = torch.device('cpu')
        torch_dtype = torch.float32
        print("Using CPU (slower)")

    # TOKENIZER AND MODELS

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast = True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    model_kl = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        torch_dtype=torch_dtype,
        trust_remote_code=True
    ).to(device)
    print(f"  Model vocab size: {model_kl.config.vocab_size}")
    print(f"  Model dtype: {torch_dtype}")
    print(f"  Model device: {device}")
    if torch_dtype == torch.float16:
        print("Gradient checkpointing disabled for float16 stability")
    else:
        try:
            if hasattr(model_kl, 'gradient_checkpointing_enable'):
                model_kl.gradient_checkpointing_enable()
                print("Gradient checkpointing enabled")
        except Exception as e:
            print(f"  Warning: Could not enable gradient checkpointing: {e}")
    print("  Testing model with dummy input...")
    try:
        test_input_ids = torch.randint(0, model_kl.config.vocab_size, (1, 10), device=device)
        with torch.no_grad():
            test_output = model_kl(test_input_ids)
            if torch.isnan(test_output.logits).any():
                print(f"  ERROR: Model produces NaN even with simple input!")
                print(f"  This suggests a model initialization issue.")
                raise ValueError("Model initialization problem - NaN in test forward pass")
            else:
                print(f"Model test forward pass successful")
    except Exception as e:
        print(f"  ERROR: Model test failed: {e}")
        raise
    reference_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    reference_model.to('cpu')  # Explicitly move to CPU
    for p in reference_model.parameters():
        p.requires_grad = False
    reference_model.eval()
    print("Reference model loaded and frozen on CPU")
    train_dataset = load_dataset(
        args.dataset_name,
        split=f"train[:{args.number_of_training_samples}]"
    )

    def validate_tokenization(text, tokenizer, max_length=MAX_LENGTH):
        """Validate that tokenization works correctly for a single text."""
        issues = []
    
    
        if not text or len(text.strip()) == 0:
            issues.append("Empty text")
            return issues
        try:
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None
            )
        except Exception as e:
            issues.append(f"Tokenization failed: {e}")
            return issues
        if 'input_ids' not in encoded:
            issues.append("Missing 'input_ids' in tokenization output")
            return issues
    
        input_ids = encoded['input_ids']
        if len(input_ids) == 0:
            issues.append("Empty token sequence")
            return issues
        vocab_size = len(tokenizer)
        invalid_ids = [idx for idx in input_ids if idx < 0 or idx >= vocab_size]
        if invalid_ids:
            issues.append(f"Invalid token IDs: {invalid_ids[:5]} (showing first 5)")
        if tokenizer.pad_token_id is not None:
            pad_count = input_ids.count(tokenizer.pad_token_id)
            if pad_count > 0:
                issues.append(f"Found {pad_count} padding tokens (should be 0 with padding=False)")
        try:
            decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
            re_encoded = tokenizer(decoded, add_special_tokens=False)['input_ids']
        except Exception as e:
            issues.append(f"Round-trip decode/encode failed: {e}")
    
        return issues

    def tokenize_function(examples):
        if 'text' in examples:
            texts = examples['text']
        elif 'prompt' in examples:
            texts = examples['prompt']
        elif 'instruction' in examples:
            texts = [
                f"{inst}\n{inp if inp else ''}" 
                for inst, inp in zip(
                    examples['instruction'], examples.get('input', ['']*len(examples['instruction']))
                )
            ]
        else:
            raise ValueError(f"Unknown dataset columns: {examples.keys()}")
        encoded = tokenizer(
            texts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,  # DataCollator handles padding
            return_tensors=None
        )
        num_to_check = min(3, len(texts))
        for i in range(num_to_check):
            text = texts[i]
            if isinstance(text, dict):
                if 'text' in text:
                    text = text['text']
                elif 'prompt' in text:
                    text = text['prompt']
                else:
                    text = str(text)
            text = str(text) if text else ""
        
            issues = validate_tokenization(text, tokenizer, MAX_LENGTH)
            if issues:
                print(f"Warning: Tokenization issues for text {i}: {issues}")
    
        return encoded
    print("TOKENIZATION VALIDATION")
    # Test tokenizer configuration
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"Tokenizer model max length: {tokenizer.model_max_length}")
    print(f"Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    print(f"UNK token: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")
    print(f"Padding side: {tokenizer.padding_side}")

    #TOKENIZATION

    if len(train_dataset) > 0:
        sample = train_dataset[0]
        sample_text = None
        if 'text' in sample:
            sample_text = sample['text']
        elif 'prompt' in sample:
            sample_text = sample['prompt'] if isinstance(sample['prompt'], str) else sample['prompt'].get('text', '')
        elif 'instruction' in sample:
            sample_text = f"{sample['instruction']}\n{sample.get('input', '')}"
    
        if sample_text:
            print(f"Sample text (first 100 chars): '{sample_text[:100]}...'")
            issues = validate_tokenization(sample_text, tokenizer, MAX_LENGTH)
            if issues:
                print(f"  ✗ Issues found: {issues}")
            else:
                encoded = tokenizer(sample_text, truncation=True, max_length=MAX_LENGTH, padding=False)
                print(f"   Tokenized successfully: {len(encoded['input_ids'])} tokens")
                print(f"    Token IDs range: [{min(encoded['input_ids'])}, {max(encoded['input_ids'])}]")
                print(f"    All token IDs valid: {all(0 <= idx < len(tokenizer) for idx in encoded['input_ids'])}")

    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing"
    )
    print(f" Tokenized {len(tokenized_train)} samples")

    # Validate tokenized dataset
    print("\nValidating tokenized dataset...")
    if len(tokenized_train) > 0:
        sample = tokenized_train[0]
        if 'input_ids' in sample:
            input_ids = sample['input_ids']
            print(f"  Sample input_ids length: {len(input_ids)}")
            print(f"  Token IDs range: [{min(input_ids)}, {max(input_ids)}]")
            print(f"  Vocab size: {len(tokenizer)}")
        
            invalid_ids = [idx for idx in input_ids if idx < 0 or idx >= len(tokenizer)]
            if invalid_ids:
                print(f"Found {len(invalid_ids)} invalid token IDs: {invalid_ids[:10]}")
            else:
                print(f"All token IDs are valid")
        
            # Check for empty sequences
            empty_count = sum(1 for item in tokenized_train if len(item.get('input_ids', [])) == 0)
            if empty_count > 0:
                print(f"   Found {empty_count} empty sequences")
            else:
                print(f"   No empty sequences")
        
            # Check sequence lengths
            lengths = [len(item.get('input_ids', [])) for item in tokenized_train]
            print(f"  Sequence length stats: min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/len(lengths):.1f}")
        else:
            print(f"   Missing 'input_ids' in tokenized dataset")
    else:
        print(f"   Tokenized dataset is empty")

    print("\nSplitting into train/validation sets...")
    train_val_split = tokenized_train.train_test_split(test_size=0.1, seed=42)
    train_dataset_tokenized = train_val_split["train"]
    val_dataset_tokenized = train_val_split["test"]
    print(f" Train: {len(train_dataset_tokenized)} samples, Val: {len(val_dataset_tokenized)} samples")

    # Final validation of split datasets
    print("\nFinal validation of tokenized datasets...")
    for name, dataset in [("Train", train_dataset_tokenized), ("Val", val_dataset_tokenized)]:
        if len(dataset) > 0:
            sample = dataset[0]
            if 'input_ids' in sample:
                input_ids = sample['input_ids']
                vocab_size = len(tokenizer)
                invalid_ids = [idx for idx in input_ids if idx < 0 or idx >= vocab_size]
                if invalid_ids:
                    print(f"   {name}: Found invalid token IDs in first sample")
                else:
                    print(f"   {name}: First sample has {len(input_ids)} valid tokens")
            else:
                print(f"   {name}: Missing 'input_ids' in samples")
        else:
            print(f"   {name}: Dataset is empty")

# =======================
# KL REGULARIZED TRAINER
# =======================
    class KLRegularizedTrainer(Trainer):
        def __init__(self, reference_model=None, kl_weight=0.1, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.reference_model = reference_model
            self.kl_weight = kl_weight

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            # Ensure inputs are on the correct device
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
            # Validate input_ids before forward pass
            input_ids = inputs["input_ids"]
            vocab_size = model.config.vocab_size
        
            # Check for invalid token IDs
            invalid_tokens = (input_ids < 0) | (input_ids >= vocab_size)
            if invalid_tokens.any():
                print(f"ERROR: Invalid token IDs found!")
                print(f"  Vocab size: {vocab_size}")
                print(f"  Input IDs shape: {input_ids.shape}")
                print(f"  Invalid token count: {invalid_tokens.sum().item()}")
                print(f"  Min ID: {input_ids.min().item()}, Max ID: {input_ids.max().item()}")
                print(f"  Input IDs sample: {input_ids[0, :20].tolist()}")
                # Clamp invalid tokens to valid range
                input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
                inputs["input_ids"] = input_ids
        
            # Check attention mask if present
            if "attention_mask" in inputs:
                attention_mask = inputs["attention_mask"]
                if torch.isnan(attention_mask).any() or (attention_mask < 0).any() or (attention_mask > 1).any():
                    print(f"WARNING: Invalid attention mask detected, fixing...")
                    inputs["attention_mask"] = attention_mask.clamp(0, 1)
        
            # Get labels - DataCollatorForLanguageModeling should provide them
            if "labels" not in inputs or inputs["labels"] is None:
                # Create labels from input_ids (for causal LM, labels = input_ids shifted)
                labels = inputs["input_ids"].clone()
            else:
                labels = inputs["labels"]
        
            model.train()
        
            # Forward pass through trainable model
            try:
                outputs = model(**inputs)
                logits = outputs.logits  # [batch, seq_len, vocab_size]
            except Exception as e:
                print(f"ERROR: Forward pass failed!")
                print(f"  Error: {e}")
                print(f"  Input IDs shape: {input_ids.shape}")
                print(f"  Input IDs range: [{input_ids.min().item()}, {input_ids.max().item()}]")
                print(f"  Vocab size: {vocab_size}")
                if "attention_mask" in inputs:
                    print(f"  Attention mask shape: {inputs['attention_mask'].shape}")
                raise
        
            # Check for NaN in logits before computing loss
            if torch.isnan(logits).any():
                print(f"ERROR: NaN detected in model logits!")
                print(f"  Logits shape: {logits.shape}")
                print(f"  NaN count: {torch.isnan(logits).sum().item()}")
                print(f"  Logits stats: min={logits.min().item():.4f}, max={logits.max().item():.4f}, mean={logits.mean().item():.4f}, std={logits.std().item():.4f}")
                print(f"  Input IDs shape: {input_ids.shape}")
                print(f"  Input IDs (first 20): {input_ids[0, :20].tolist()}")
                print(f"  Vocab size: {vocab_size}")
                print(f"  Model dtype: {logits.dtype}")
                print(f"  Model device: {logits.device}")
            
                # Check if it's a float16 issue
                if logits.dtype == torch.float16:
                    print(f"  Attempting to diagnose float16 issue...")
                    # Check if input embeddings might be the issue
                    if hasattr(model, 'get_input_embeddings'):
                        emb = model.get_input_embeddings()
                        if emb is not None:
                            print(f"  Embedding weight stats: min={emb.weight.min().item():.4f}, max={emb.weight.max().item():.4f}")
                            if torch.isnan(emb.weight).any():
                                print(f"  ERROR: NaN in embedding weights!")
            
                raise ValueError("NaN in model logits - check model initialization and input data")
        
            # Shift for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
        
            # Check label validity
            valid_labels = (shift_labels != -100)
            num_valid = valid_labels.sum().item()
        
            if num_valid == 0:
                print(f"ERROR: No valid labels found!")
                print(f"  Labels shape: {labels.shape}")
                print(f"  Shift labels shape: {shift_labels.shape}")
                print(f"  All labels are -100 (padding)")
                print(f"  Input IDs (first 20): {inputs['input_ids'][0, :20].tolist()}")
                print(f"  Labels (first 20): {labels[0, :20].tolist()}")
                raise ValueError("All labels are -100 - check data collator and tokenization")
        
            # Check for invalid label values (outside vocab range)
            vocab_size = logits.size(-1)
            invalid_labels = (shift_labels >= vocab_size) | (shift_labels < 0)
            invalid_labels = invalid_labels & valid_labels  # Only check non-ignored labels
        
            if invalid_labels.any():
                print(f"ERROR: Invalid label values found!")
                print(f"  Vocab size: {vocab_size}")
                print(f"  Invalid label count: {invalid_labels.sum().item()}")
                print(f"  Max label: {shift_labels[valid_labels].max().item()}")
                print(f"  Min label: {shift_labels[valid_labels].min().item()}")
                # Clamp invalid labels to valid range
                shift_labels = torch.clamp(shift_labels, 0, vocab_size - 1)
        
            # Standard cross-entropy loss
            if logits.dtype == torch.float16:
                shift_logits_fp32 = shift_logits.float()
            else:
                shift_logits_fp32 = shift_logits
        
            ce_loss = F.cross_entropy(
                shift_logits_fp32.view(-1, shift_logits_fp32.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction='mean'
            )
        
            if logits.dtype == torch.float16:
                ce_loss = ce_loss.half()
        
            # Check if loss is valid
            if torch.isnan(ce_loss) or torch.isinf(ce_loss):
                print(f"ERROR: Invalid CE loss: {ce_loss.item()}")
                print(f"  Input shape: {inputs['input_ids'].shape}")
                print(f"  Labels shape: {labels.shape}")
                print(f"  Valid labels: {num_valid} / {shift_labels.numel()}")
                print(f"  Logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
                print(f"  Logits mean: {logits.mean().item():.2f}, std: {logits.std().item():.2f}")
                raise ValueError(f"NaN/Inf in CE loss computation")

            if self.reference_model is not None:
                # Move inputs to reference model device (CPU)
                inputs_ref = {k: v.detach().cpu() for k, v in inputs.items()}
            
                with torch.no_grad():
                    ref_outputs = self.reference_model(**inputs_ref)
                    ref_logits = ref_outputs.logits.to(model.device).float()
            
                # Check for NaN in reference logits
                if torch.isnan(ref_logits).any():
                    print(f"ERROR: NaN in reference model logits!")
                    raise ValueError("NaN in reference model logits")
            
                shift_ref_logits = ref_logits[..., :-1, :].contiguous()
            
                # Create valid mask (ignore padding tokens)
                valid_mask = (shift_labels != -100).float()
                num_valid_tokens = valid_mask.sum().clamp(min=1)
            
                # Compute KL divergence: KL(reference || model)
                # Use float32 for numerical stability
                if shift_logits.dtype == torch.float16:
                    shift_logits_fp32 = shift_logits.float()
                    shift_ref_logits_fp32 = shift_ref_logits.float()
                else:
                    shift_logits_fp32 = shift_logits
                    shift_ref_logits_fp32 = shift_ref_logits
            
                log_probs = F.log_softmax(shift_logits_fp32, dim=-1)  # log(model_probs)
                ref_probs = F.softmax(shift_ref_logits_fp32, dim=-1)   # ref_probs (not log)
            
                # Add small epsilon to avoid log(0)
                ref_probs = ref_probs + 1e-8
                ref_probs = ref_probs / ref_probs.sum(dim=-1, keepdim=True)  # Renormalize
            
           
                kl_per_token = F.kl_div(
                    log_probs,  # input: log probabilities of model
                    ref_probs,  # target: probabilities of reference
                    reduction="none"
                ).sum(dim=-1)  # Sum over vocabulary dimension
            
                # Apply valid mask and average
                kl_per_token = kl_per_token * valid_mask
                kl_loss = kl_per_token.sum() / num_valid_tokens
            
                if shift_logits.dtype == torch.float16:
                    kl_loss = kl_loss.half()
            
                # Check for NaN in KL loss
                if torch.isnan(kl_loss) or torch.isinf(kl_loss):
                    print(f"ERROR: Invalid KL loss: {kl_loss.item()}")
                    print(f"  Valid tokens: {num_valid_tokens.item()}")
                    raise ValueError(f"NaN/Inf in KL loss computation")
            
                # Final loss
                loss = ce_loss + self.kl_weight * kl_loss
            
                # Final check
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"ERROR: Invalid total loss: {loss.item()}")
                    print(f"  CE loss: {ce_loss.item():.4f}")
                    print(f"  KL loss: {kl_loss.item():.4f}")
                    raise ValueError(f"NaN/Inf in total loss")
            else:
                loss = ce_loss

            return (loss, outputs) if return_outputs else loss
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,  # Causal language modeling
        pad_to_multiple_of=None,
        return_tensors="pt"  # Ensure PyTorch tensors
    )


    # TRAINING ARGUMENTS

    training_args_kl = TrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="no",
    load_best_model_at_end=False,
    fp16=(device.type=="cuda" and not FORCE_CPU and torch_dtype == torch.float16),
    no_cuda=(device.type=="cpu" or FORCE_CPU),
    report_to="none",
    save_total_limit=1,
)

    # Trainer
    trainer_kl = KLRegularizedTrainer(
        model=model_kl,
        args=training_args_kl,
        train_dataset=train_dataset_tokenized,
        eval_dataset=val_dataset_tokenized,
        data_collator=data_collator,
        reference_model=reference_model,
        kl_weight=KL_WEIGHT,
    )


    # FINE-TUNING

    print("\n" + "="*70)
    print("Starting KL-regularized fine-tuning...")
    print("="*70)
    print(f"Train samples: {len(train_dataset_tokenized)}")
    print(f"Val samples: {len(val_dataset_tokenized)}")
    print(f"KL weight: {KL_WEIGHT}")
    print(f"Device: {device}")
    print(f"Model dtype: {torch_dtype}")
    print("="*70 + "\n")

    # Verify dataset format
    if len(train_dataset_tokenized) > 0:
        sample = train_dataset_tokenized[0]
        print(f"Sample keys: {sample.keys()}")
        if 'input_ids' in sample:
            print(f"Input IDs shape: {len(sample['input_ids'])}")
            print(f"Input IDs (first 20): {sample['input_ids'][:20]}")
            print(f"Input IDs (last 20): {sample['input_ids'][-20:]}")
    
        # Test data collator
        print("\nTesting data collator...")
        test_batch = data_collator([train_dataset_tokenized[0]])
        print(f"Collated batch keys: {test_batch.keys()}")
        if 'input_ids' in test_batch:
            input_ids = test_batch['input_ids']
            print(f"Batch input_ids shape: {input_ids.shape}")
            print(f"Input IDs range: [{input_ids.min().item()}, {input_ids.max().item()}]")
            print(f"Input IDs (first 20): {input_ids[0, :20].tolist()}")
        
            # Check against vocab size
            vocab_size = tokenizer.vocab_size
            print(f"Tokenizer vocab size: {vocab_size}")
            invalid_ids = (input_ids < 0) | (input_ids >= vocab_size)
            if invalid_ids.any():
                    print(f"WARNING: Invalid token IDs found: {invalid_ids.sum().item()} tokens")
            else:
                print(f"All token IDs are valid")
        
            print(f"Batch labels shape: {test_batch['labels'].shape if 'labels' in test_batch else 'No labels'}")
            if 'labels' in test_batch:
                labels = test_batch['labels']
                valid_labels = (labels != -100)
                num_valid = valid_labels.sum().item()
                total_labels = labels.numel()
                print(f"Valid labels: {num_valid} / {total_labels} ({100*num_valid/total_labels:.1f}%)")
                if num_valid > 0:
                    valid_label_values = labels[valid_labels]
                    print(f"Label range: [{valid_label_values.min().item()}, {valid_label_values.max().item()}]")
                    invalid_label_ids = (valid_label_values < 0) | (valid_label_values >= vocab_size)
                    if invalid_label_ids.any():
                        print(f"WARNING: Invalid label values found: {invalid_label_ids.sum().item()} labels")
                    else:
                        print(f"All label values are valid")
                else:
                    print(f"ERROR: No valid labels found!")
        print()

    try:
        trainer_kl.train()
        trainer_kl.save_model(args.output_dir)
        print(f"\n KL-regularized model saved to {args.output_dir}")
    except Exception as e:
        print(f"\n✗ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise
if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Standard finetuning")
    parser.add_argument('--model_name', type=str, description='Pass the model name')
    parser.add_argument('--dataset_name',type=str, description='Pass the HF dataset name' )
    parser.add_argument('--number_of_training_samples', type=int, description='No of training samples')
    parser.add_argument('--output_dir', type=str, description='Path to output directory')
    args = parser.parse_args()
    main(args)

