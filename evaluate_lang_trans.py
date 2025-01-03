from transformers import AutoTokenizer,AutoModelForSeq2SeqLM, pipeline, DataCollatorForSeq2Seq
# import language_translation as lt
import torch
import evaluate

model = AutoModelForSeq2SeqLM.from_pretrained("./4_lora_fine_tuned_t5_translation_model")
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base", padding_side="left")
tokenizer.pad_token = tokenizer.eos_token = '<pad>'

input_ids = tokenizer("translate English to Finnish: This is a sad line", return_tensors="pt").input_ids
outputs = model.generate(input_ids, max_new_tokens = 512)
output = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(output)

model = AutoModelForSeq2SeqLM.from_pretrained("./9_lora_fine_tuned_t5_translation_model")
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base", padding_side="left")
tokenizer.pad_token = tokenizer.eos_token = '<pad>'

input_ids = tokenizer("translate English to Finnish: this is a sad line", return_tensors="pt").input_ids
outputs = model.generate(input_ids, max_new_tokens = 512)
output = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(output)

# input_ids = tokenizer("How are you today?", return_tensors="pt").input_ids
# outputs = model.generate(input_ids, max_new_tokens = 512)
# output = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(output)

# input_ids = tokenizer("translate English to French: How are you today?", return_tensors="pt").input_ids
# outputs = model.generate(input_ids, max_new_tokens = 512)
# output = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(output)

# tokenized_prompts = tokenizer(['Is this', 'What is the matter'], padding=True, return_tensors='pt')
# set_seed(42)
# model_op = model.generate(input_ids=tokenized_prompts['input_ids'].to(device),
#                           attention_mask=tokenized_prompts['attention_mask'].to(device),
#                           renormalize_logits=False, do_sample=True,
#                           use_cache=True, max_new_tokens=10)
# tokenizer.batch_decode(model_op, skip_special_tokens=True)
# inputs = torch.tensor(["Hello, this is a test", "The quick brown fox jumped above"])
# outtput_tokenized = lt.tokenizer()
# outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)
# print(outputs)

# translator = pipeline("translation_en_to_fi", model="./lora_fine_tuned_t5_translation_model", tokenizer="./lora_fine_tuned_t5_translation_model", device="cuda")
# result = translator("Translate English to Finnish: How are you?")
# print(result)


# bleu = evaluate.load("bleu")
# results = bleu.compute(predictions=output, references="Talo on ihana.")
# print(results)
# Das Haus ist wunderbar.

# translator = pipeline("translation_en_to_fi",model='google-t5/t5-base', device="cuda")
# print("Pipeline base model translation: \n")
# print(translator("translate English to Finnish: The house is wonderful."))

# from transformers import T5Tokenizer, T5ForConditionalGeneration

# tokenizer = T5Tokenizer.from_pretrained('t5-small')

# model = T5ForConditionalGeneration.from_pretrained('t5-small', return_dict=True)

# input = "My name is quack quack and I live in the north pole"
# print("Translate using t5 small")
# # You can also use "translate English to French" and "translate English to Romanian"
# input_ids = tokenizer("translate English to Finnish: "+input, return_tensors="pt").input_ids  # Batch size 1

# outputs = model.generate(input_ids)

# decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(decoded)