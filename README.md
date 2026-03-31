[![License](LICENSE.txt)](LICENSE.txt)

This repository contains the codes for the term paper of the seminar "LLM-Safety" where we propose a refusal aware finetuning method to preserve refusal awareness of the finetuned models.

## Models Used
In the --model_name field below, one may choose any one of the following three models which we use for our experiments from huggingface (🤗): 
```google/gemma-3-270m-it```, ```h2oai/h2o-danube3-500m-chat``` and ```HuggingFaceTB/SmolLM-135M-Instruct```

## Training

## Standard Finetuning
First we train all the models on the benign dataset without updating the objective function.
```bash
python standard_finetuning.py \
  --model_name "h2oai/h2o-danube3-500m-chat"\
  --dataset_name "LLM-LAT/benign-dataset" \
  --number_of_training_samples 50 \
  --output_dir "./finetuned_standard" 
```
## Token level Finetuning
We next all the models using the taks objective updated by a regularizer defined in terms of KL divergence at the token level distribution of the base model and the model being finetuned
```bash
python finetuning_with_token-level-kl.py \
  --model_name "h2oai/h2o-danube3-500m-chat"\
  --dataset_name "LLM-LAT/benign-dataset" \
  --number_of_training_samples 50 \
  --output_dir "./finetuned_kl_regularized" 
```
## Subspace level finetuning
We finally all the models using the taks objective updated by our proposed Refusal Space Drift Regularization (RSDR) method
```bash
python subspace_preserving_finetune.py \
  --model_name "h2oai/h2o-danube3-500m-chat"\
  --dataset_name "LLM-LAT/benign-dataset" \
  --number_of_training_samples 50 \
  --output_dir "./subspace_preserving_finetuned" 
  ```
## Evaluation
## Prerequisites

We need the Groq API to perform inference:
```bash
groq_api_key = "groq_api_key"
```
```bash
python LLM_as_judge.py \
  --base_model "h2oai/h2o-danube3-500m-base" \
  --standard_finetune "./finetuned_standard" \
  --token_kl_finetune "./finetuned_kl_regularized" \
  --subspace_kl_finetune "./subspace_preserving_finetuned" \
  --dataset_name "allenai/real-toxicity-prompts" \
  --number_of_training_samples 5000  \
  --output_dir "./eval_results" 
  --results_file "llm_judge_results.json" \
  --summary_file "refusal_summary.json" \
  --overlap_file  "overlap_analysis.json"
```

### **Contact**:
```bib
Saugata Purkayastha
Email: sapu00001@stud.uni-saarland.de
```
