from datasets import load_dataset
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM, pipeline, DataCollatorForSeq2Seq, T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from wordcloud import WordCloud
import matplotlib.pyplot as plt

  

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")

source_lang = "en"
target_lang = "fi"
# Step 2: Load and preprocess the dataset (Helsinki-NLP/tatoeba)
# Specify the language pair (e.g., English to French: 'eng-fra')
# language_pair = "eng-fra"
# raw_datasets = load_dataset("Helsinki-NLP/tatoeba", lang1=source_lang, lang2=target_lang , trust_remote_code=True)
# print(raw_datasets['train'])
raw_dataset_2 = load_dataset("Helsinki-NLP/opus_books", source_lang +"-"+target_lang , split='train')
# print(raw_dataset_2['translation'][source_lang])

# max_len_src = 0
# max_len_tgt = 0

# for item in raw_datasets["train"]:
#     # print(item['translation'][source_lang])
#     # break
#     src_ids = tokenizer.encode(item['translation'][source_lang])
#     tgt_ids = tokenizer.encode(item['translation'][target_lang])
#     max_len_src = max(max_len_src, len(src_ids))
#     max_len_tgt = max(max_len_tgt, len(tgt_ids))

# print(f'Max length of train source sentence: {max_len_src}')
# print(f'Max length of train target sentence: {max_len_tgt}')

# dataset_en = ""
# for item in raw_datasets["train"]:
#     dataset_en += " " + item['translation'][source_lang]
# word_cloud = WordCloud().generate(dataset_en)
# plt.imshow(word_cloud)
# plt.show()

# dataset_fi = ""
# for item in raw_datasets["train"]:
#     dataset_fi += " " + item['translation'][target_lang]
# word_cloud = WordCloud().generate(dataset_fi)
# plt.imshow(word_cloud)

# plt.show()

max_len_src = 0
max_len_tgt = 0

for item in raw_dataset_2:
    print(item['translation'][source_lang])
    break
    src_ids = tokenizer.encode(item['translation'][source_lang])
    tgt_ids = tokenizer.encode(item['translation'][target_lang])
    max_len_src = max(max_len_src, len(src_ids))
    max_len_tgt = max(max_len_tgt, len(tgt_ids))

# print(f'Max length of train source sentence: {max_len_src}')
# print(f'Max length of train target sentence: {max_len_tgt}')

# dataset_en = ""
# for item in raw_datasets:
#     dataset_en += " " + item['translation'][source_lang]
# word_cloud = WordCloud().generate(dataset_en)
# plt.imshow(word_cloud)
# plt.show()

# dataset_fi = ""
# for item in raw_datasets:
#     dataset_fi += " " + item['translation'][target_lang]
# word_cloud = WordCloud().generate(dataset_fi)
# plt.imshow(word_cloud)

# plt.show()
