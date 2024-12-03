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

# Step 1: Load the pre-trained T5 model and tokenizer
model_name = "t5-small"  # Choose T5 variant like 't5-base' or 't5-large' for larger models
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")  

source = 'en'
target = 'fi'
# Step 2: Load and preprocess the dataset (Helsinki-NLP/tatoeba)
# Specify the language pair (e.g., English to French: 'eng-fra')
# language_pair = "eng-fra"
raw_datasets = load_dataset("Helsinki-NLP/tatoeba", lang1='en', lang2='fi' , trust_remote_code=True)
print(raw_datasets['train'])
split_datasets = raw_datasets["train"].train_test_split(train_size=0.1, seed=20)
print(split_datasets)
split_datasets = split_datasets["train"].train_test_split(train_size=0.7, seed=20)
print(split_datasets)

split_datasets["validation"] = split_datasets.pop("test")
print(split_datasets["train"][1]["translation"])

translator = pipeline("translation_en_to_fi", model=model_name, device="cuda")
print(translator("Default to expanded threads"))

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
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


# Step 4: Define evaluation metrics
bleu = evaluate.load("bleu")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Format labels for BLEU
    decoded_labels = [[label] for label in decoded_labels]
    results = bleu.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": results["bleu"]}

lora_config = LoraConfig(
    r=32, # Rank
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM" # FLAN-T5
)
peft_model = get_peft_model(model, lora_config)


# Step 5: Define training arguments
training_args = Seq2SeqTrainingArguments(
    # peft_model,
    output_dir="./lora_t5_translation_model",
    evaluation_strategy="epoch",
    learning_rate=1e-5,
    # per_device_train_batch_size=16,
    # per_device_eval_batch_size=16,
    # weight_decay=0.01,
    # save_total_limit=3,
    # num_train_epochs=3,
    predict_with_generate=True,
    fp16=True,  # Enable if GPU supports mixed precision
    logging_dir="./lora_finetune_logs",
    logging_steps=10,
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
trainer.train()

# Step 8: Save the fine-tuned model
trainer.save_model("./lora_fine_tuned_t5_translation_model")
tokenizer.save_pretrained("./lora_fine_tuned_t5_translation_model")

# Step 9: Test the fine-tuned model
from transformers import pipeline

translator = pipeline("translation", model="./lora_fine_tuned_t5_translation_model", tokenizer="./lora_fine_tuned_t5_translation_model")
result = translator("How are you?")
print(result)
