# # from transformers import T5ForConditionalGeneration, T5Tokenizer

# # model_name = 'jbochi/madlad400-3b-mt'
# # model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")
# # tokenizer = T5Tokenizer.from_pretrained(model_name)

# # text = "<2pt> I love pizza!"
# # input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
# # outputs = model.generate(input_ids=input_ids)

# # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# # # Eu adoro pizza!



# # Use a pipeline as a high-level helper
# from transformers import pipeline
# import torch 

# pipe = pipeline("translation_en_to_fr", model="google-t5/t5-large", device= torch.device("cuda"))
# print("THis is a test english sentence")
# print("*****")
# print(pipe("I you are?"))




# from datasets import load_dataset
# from transformers import pipeline
# from transformers import AutoTokenizer


# raw_datasets = load_dataset("kde4", lang1="en", lang2="fr")

# print(raw_datasets)

# split_datasets = raw_datasets["train"].train_test_split(train_size=0.9, seed=20)
# print(split_datasets)

# split_datasets["validation"] = split_datasets.pop("test")
# split_datasets["train"][1]["translation"]


# model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
# translator = pipeline("translation", model=model_checkpoint)
# print(translator("Default to expanded threads"))

# # print(split_datasets["train"][172]["translation"])


# model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, return_tensors="pt")
# en_sentence = split_datasets["train"][1]["translation"]["en"]
# fr_sentence = split_datasets["train"][1]["translation"]["fr"]

# inputs = tokenizer(en_sentence, text_target=fr_sentence)
# inputs




# Install necessary libraries if not already installed
# pip install transformers datasets evaluate torch

from transformers import AutoTokenizer,AutoModelForSeq2SeqLM, pipeline, DataCollatorForSeq2Seq, T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
import evaluate
import torch
from peft import LoraConfig, get_peft_model
import numpy as np
import os
import torchmetrics

# 12:40 am logs

# Step 1: Load the pre-trained T5 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")  

source = 'en'
target = 'fi'
# Step 2: Load and preprocess the dataset (Helsinki-NLP/tatoeba)
# Specify the language pair (e.g., English to French: 'eng-fra')
# language_pair = "eng-fra"
raw_datasets = load_dataset("Helsinki-NLP/tatoeba", lang1='en', lang2='fi' , trust_remote_code=True)
# print(raw_datasets['train'])
split_dataset = raw_datasets["train"].train_test_split(train_size=0.7, seed=20)
print(split_dataset) 
#Giving the model 70% from 60% dataset from epoch 100, which weirdly made the epoch count to 70.
split_datasets = split_dataset["train"].train_test_split(train_size=0.7, seed=20)
print(split_datasets)

split_datasets["validation"] = split_datasets.pop("test")
# print(split_datasets["train"][1]["translation"])

# model_name = "t5-small"  # Choose T5 variant like 't5-base' or 't5-large' for larger models
# translator = pipeline("translation_en_to_fi", model=model_name, device="cuda")
# print(translator("Default to expanded threads"))

source_lang = "en"
target_lang = "fi"
prefix = "translate English to Finnish: "

# Preprocessing function
def preprocess_function(examples):
    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs

# # Tokenize the dataset
# tokenized_datasets = raw_datasets['train'].map(preprocess_function, batched=True)
tokenized_datasets = split_datasets.map(preprocess_function, batched=True)


# print(tokenized_datasets)
# # print(tokenized_datasets["input_ids"])
# # print(tokenized_datasets["labels"])


# Step 3: Create data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=tokenizer.pad_token_id)


# Step 4: Define evaluation metrics
bleu = evaluate.load("bleu")
bleu_metric = torchmetrics.BLEUScore()

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    # Replace -100s in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Format labels for BLEU
    decoded_labels = [[label] for label in decoded_labels]
    # print(decoded_labels)
    results = bleu.compute(predictions=decoded_preds, references=decoded_labels)
    bleu_2 = bleu_metric(decoded_preds, decoded_labels)
    # print("bleu22222")
    # print(bleu_2)
    print(results)
    return {"bleu": results["bleu"],
            "bleu_2": bleu_2
    }

lora_config = LoraConfig(
    r=32, # Rank
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM" # FLAN-T5
)
peft_model = get_peft_model(model, lora_config)

save_directory = "./4_lora_t5_translation_model"
# Step 5: Define training arguments
training_args = Seq2SeqTrainingArguments(
    # peft_model,
    output_dir= save_directory,
    evaluation_strategy="epoch",
    learning_rate=2e-5, # changed learning rate at epoch 100
    per_device_train_batch_size=20, # turned on batch size at epoch 80
    per_device_eval_batch_size=20,
    weight_decay=0.01, #changed from 0.01 to 0.1 at epoch 48.5
    save_total_limit=3, # enabled at 100
    num_train_epochs=150,
    predict_with_generate=True,
    # fp16=True,  # Enable if GPU supports mixed precision
    logging_dir="./4_lora_finetune_logs",
    logging_steps=10,
)
# if os.path.isdir("./lora_t5_translation_model"):
#     train_model = AutoModelForSeq2SeqLM.from_pretrained("./lora_fine_tuned_t5_translation_model")  

#     print("Using previously fine tuned model.........")
# else:
#     train_model = peft_model
#     print("Starting from scracth.........")
    
# Step 6: Initialize Trainer
trainer = Seq2SeqTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


# Step 7: Fine-tune the model

# Check for a checkpoint to resume training
checkpoint_path = os.path.join(save_directory, "checkpoint-429000") # Change before training
print(checkpoint_path)
if os.path.exists(checkpoint_path):
    print("Resuming training from the last checkpoint...")
    trainer.train(resume_from_checkpoint=checkpoint_path)
else:
    print("Starting training from scratch...")
    trainer.train()
# trainer.train()

# Step 8: Save the fine-tuned model
trainer.save_model("./4_lora_fine_tuned_t5_translation_model")
tokenizer.save_pretrained("./4_lora_fine_tuned_t5_translation_model")

# Step 9: Test the fine-tuned model
# from transformers import pipeline

# translator = pipeline("translation_en_to_fi", model="./4_lora_fine_tuned_t5_translation_model", 
#                       tokenizer="./4_lora_fine_tuned_t5_translation_model", device="cuda")
# result = translator("Translate English to Finnish: How are you?")
# print(result)
