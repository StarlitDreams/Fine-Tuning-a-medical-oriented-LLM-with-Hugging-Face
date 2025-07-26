import torch
from trl import SFTTrainer
from peft import LoraConfig
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments,pipeline)

llama_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path="aboonaji/llama2finetune-v2",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=getattr(torch,"float16"),
    bnb_4bit_quant_type="nf4"))
llama_model.config.use_cache = False
llama_model.config.pretraining_tp = 1

llama_tokenizer =  AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path="aboonaji/llama2finetune-v2",
    use_fast=False,trust_remote_code=True
    )
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"

TrainingArguments = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=8,
    max_steps=1000
    )

llama_sft_trainer = SFTTrainer(
    model=llama_model,
    args=TrainingArguments,
    train_dataset=load_dataset("aboonaji/wiki_medical_terms_llam2_format", split="train"),
    tokenizer=llama_tokenizer,
    pert_config=LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.01,
        task_type="CAUSAL_LM",
        ),
    dataset_text_field="text"
    )

llama_sft_trainer.train()

user_prompt = ""
text_generation_pipeline = pipeline(
    task="text-generation", 
    model=llama_model, 
    tokenizer=llama_tokenizer,
    max_lenght=300
    )

model_answer = text_generation_pipeline(f"<s>[INST]{user_prompt}[/INST]")
print(model_answer[0]['generated_text'])



