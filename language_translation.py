# from transformers import T5ForConditionalGeneration, T5Tokenizer

# model_name = 'jbochi/madlad400-3b-mt'
# model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")
# tokenizer = T5Tokenizer.from_pretrained(model_name)

# text = "<2pt> I love pizza!"
# input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
# outputs = model.generate(input_ids=input_ids)

# print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# # Eu adoro pizza!



# Use a pipeline as a high-level helper
from transformers import pipeline
import torch 

pipe = pipeline("translation_en_to_fr", model="google-t5/t5-large", device= torch.device("cuda"))
print("THis is a test english sentence")
print("*****")
print(pipe("I you are?"))