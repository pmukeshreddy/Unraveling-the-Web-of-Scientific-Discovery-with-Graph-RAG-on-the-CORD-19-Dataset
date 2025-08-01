from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from peft import LoraConfig, get_peft_model
import pandas as pd

# Load dataset (from JSON if saved, or raft_data list)
df = pd.read_json('raft_dataset.json')
dataset = Dataset.from_pandas(df)

# Format as prompts (question + docs → answer)
def format_prompt(ex):
    docs = "\n".join(ex['documents'])
    return {"text": f"Question: {ex['question']}\nDocuments:\n{docs}\nAnswer: {ex['answer']}"}

dataset = dataset.map(format_prompt)

# Model/tokenizer
model_name = "gpt2"  # Or "meta-llama/Llama-2-7b-hf" with HF token
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # For padding
model = AutoModelForCausalLM.from_pretrained(model_name)

# Tokenize
def tokenize(ex):
    return tokenizer(ex['text'], truncation=True, padding='max_length', max_length=512)

tokenized_dataset = dataset.map(tokenize, batched=True)
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2)

# LoRA config
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=['c_attn', 'c_proj'], lora_dropout=0.05)

model = get_peft_model(model, lora_config)

# Training args
args = TrainingArguments(
    output_dir='./fine_tuned_raft',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,  # For GPU
    save_steps=500,
    eval_steps=500,
    logging_steps=100,
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test']
)

trainer.train()
trainer.save_model('./fine_tuned_raft_model')
tokenizer.save_pretrained('./fine_tuned_raft_model')
