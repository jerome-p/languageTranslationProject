from transformers import AutoTokenizer,AutoModelForSeq2SeqLM, pipeline, DataCollatorForSeq2Seq, T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
import evaluate
import torch
from peft import LoraConfig, get_peft_model
import numpy as np
import os
import torchmetrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 12:40 am logs

# Step 1: Load the pre-trained T5 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")  

# Different model families use different names for the same field
typical_fields = ["max_position_embeddings", "n_positions", "seq_len", "seq_length", "n_ctx", "sliding_window"]  

# Check which attribute a given model object has
context_windows = [getattr(model.config, field) for field in typical_fields if field in dir(model.config)]

# Grab the last one in the list; usually there's only 1 anyway
print(context_windows.pop()) if len(context_windows) else print(f"No Max input variable found for T5")

source_lang = "en"
target_lang = "fi"
# Step 2: Load and preprocess the dataset (Helsinki-NLP/tatoeba)
# Specify the language pair (e.g., English to French: 'eng-fra')
# language_pair = "eng-fra"
raw_datasets = load_dataset("Helsinki-NLP/tatoeba", lang1=source_lang, lang2=target_lang , trust_remote_code=True)
# print(raw_datasets['train'])

max_len_src = 0
max_len_tgt = 0

for item in raw_datasets["train"]:
    # print(item['translation'][source_lang])
    # break
    src_ids = tokenizer.encode(item['translation'][source_lang])
    tgt_ids = tokenizer.encode(item['translation'][target_lang])
    max_len_src = max(max_len_src, len(src_ids))
    max_len_tgt = max(max_len_tgt, len(tgt_ids))
    
print(f'Max length of train source sentence: {max_len_src}')
print(f'Max length of train target sentence: {max_len_tgt}')

# for item in raw_datasets["test"]:
#     # print(item['translation'][source_lang])
#     # break
#     src_ids = tokenizer.encode(item['translation'][source_lang])
#     tgt_ids = tokenizer.encode(item['translation'][target_lang])
#     max_len_src = max(max_len_src, len(src_ids))
#     max_len_tgt = max(max_len_tgt, len(tgt_ids))
    
# print(f'Max length of test source sentence: {max_len_src}')
# print(f'Max length of test target sentence: {max_len_tgt}')

split_dataset = raw_datasets["train"].train_test_split(train_size=0.2, seed=20)   #was 0.7
print(split_dataset) 
#Giving the model 70% from 60% dataset from epoch 100, which weirdly made the epoch count to 70.
split_datasets = split_dataset["train"].train_test_split(train_size=0.85, seed=20) #was 0.7
print(split_datasets)

split_datasets["validation"] = split_datasets.pop("test")
# print(split_datasets["train"][1]["translation"])

# model_name = "t5-small"  # Choose T5 variant like 't5-base' or 't5-large' for larger models
# translator = pipeline("translation_en_to_fi", model=model_name, device="cuda")
# print(translator("Default to expanded threads"))


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
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # labels = np.where(labels < 0, labels, tokenizer.pad_token_id)
    
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

    precision, recall, f1, _ = precision_recall_fscore_support(decoded_labels, decoded_preds, average="weighted")
    acc = accuracy_score(decoded_labels, decoded_preds)
    return {"bleu": results["bleu"],
            "bleu_2": bleu_2,
            "accuracy": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall
    }

lora_config = LoraConfig(
    r=20, # Rank # was 32
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM" # FLAN-T5
)

peft_model = get_peft_model(model, lora_config)


save_directory = "./9_lora_t5_translation_model"
# Step 5: Define training arguments
training_args = Seq2SeqTrainingArguments(
    # peft_model,
    output_dir= save_directory,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-4, # changed learning rate at epoch 100
    per_device_train_batch_size=16, # turned on batch size at epoch 80
    per_device_eval_batch_size=16,
    weight_decay=0.01, #changed from 0.01 to 0.1 at epoch 48.5
    save_total_limit=10, # enabled at 100
    num_train_epochs=50, # was 250
    predict_with_generate=True,
    generation_max_length = 80, # Since train data max length is 389
    # fp16=True,  # Enable if GPU supports mixed precision
    logging_dir="./9_lora_finetune_logs",
    logging_steps=10,
    load_best_model_at_end=True
)
    
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

# if os.path.isdir("./lora_t5_translation_model"):
#     train_model = AutoModelForSeq2SeqLM.from_pretrained("./lora_fine_tuned_t5_translation_model")  

#     print("Using previously fine tuned model.........")
# else:
#     train_model = peft_model
#     print("Starting from scracth.........")


# Check for a checkpoint to resume training
checkpoint_path = os.path.join(save_directory, "checkpoint-9000") # Change before training
print(checkpoint_path)
if os.path.exists(checkpoint_path): 
    print("Resuming training from the last checkpoint...")
    trainer.train(resume_from_checkpoint=checkpoint_path)
else:
    print("Starting training from scratch...")
    trainer.train()
# trainer.train()

# Step 8: Save the fine-tuned model
trainer.save_model("./9_lora_fine_tuned_t5_translation_model")
tokenizer.save_pretrained("./9_lora_fine_tuned_t5_translation_model_tokenizer")

# Step 9: Test the fine-tuned model
# from transformers import pipeline

# translator = pipeline("translation_en_to_fi", model="./4_lora_fine_tuned_t5_translation_model", 
#                       tokenizer="./4_lora_fine_tuned_t5_translation_model", device="cuda")
# result = translator("Translate English to Finnish: How are you?")
# print(result)
