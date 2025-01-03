from transformers import AutoTokenizer,AutoModelForSeq2SeqLM, pipeline

def evaluate_model(model_path, tokenizer_name, input, max_tokens, prefix="translate English to Finnish: "):
    """
    model_path: str
    tokenizer_name: str
    input: str
    max_tokens: int
    prefix: str
    
    This functions takes in model path, tokenizer and the text input.
    Which is then used to genrate output using the chosen model from the model_path.
    returns str
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    input_ids = tokenizer(prefix+input, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_new_tokens = max_tokens)
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print(output)
    return output



translator = pipeline("translation_en_to_fi",model='google-t5/t5-base', device="cuda", max_new_tokens=80)
print("Pipeline base model translation:")
print("EN->FR")
print(translator("translate English to French: The house is wonderful."))
print("EN->DE")
print(translator("translate English to German: The house is wonderful."))
print("EN->FI")
print(translator("translate English to Finnish: The house is wonderful."))
print("T5 base model cannot translate to Finnish")

print("Evaluating using fine tuned T5model: \n")


print(evaluate_model("./4_lora_fine_tuned_t5_translation_model", 
               "google-t5/t5-base",
               "The house is wonderful.",
               80))

model_name = "./9_lora_fine_tuned_t5_translation_model_epoch100"
print(evaluate_model(model_name, 
               "google-t5/t5-base",
               "The house is wonderful.",
               80))
print(evaluate_model(model_name, 
               "google-t5/t5-base",
               "I have mixed emotions!",
               80))
print(evaluate_model(model_name, 
               "google-t5/t5-base",
               "How are you today?",
               80))
