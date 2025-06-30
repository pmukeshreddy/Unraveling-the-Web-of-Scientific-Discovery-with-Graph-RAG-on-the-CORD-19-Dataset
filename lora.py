from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import LoraConfig , get_peft_model


dataset = load_dataset("json",data_files="cypher-finetuning-dataset.jsonl",split="train")

quantization_config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.float16)

model_id = "meta-llama/Meta-Llama-3-8B"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id,quantization_config=quantization_config,device_map="auto")



lora_config = LoraConfig(r=8,lora_alpha=16,target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM")
peft_model = get_peft_model(model,lora_config)
