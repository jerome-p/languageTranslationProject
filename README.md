# Machine Language Translation using Deep Learing methods.

This repository contains code for a language translation project.

The repo consists of 3 models 
1. Seq2Seq (broken)
2. Transformer
3. Fine-tuned T5 base model.

Running the files
- To run each model, make sure you are in model's folder.
  -  The transformer model is trained by running the main.py file.
  - To train the T5 model, run the language_translation.py file.

The Transformer model follows the implementaion by https://github.com/hkproj/pytorch-transformer
and all the configurations for traingin the model is done via the config file. Edit this file before running the main.py file

Fine tuned model, uses hugging face APIs. All the log files and model files saved by training each of the model will in saved in the model's directory.

Requuirements:
- Python version 3.9
- PyTorch (Cuda version if using GPU)
a requirements.txt file is provided, with the libraries installed on my local setup.

Visualisaions.py 
- This file expects a folder with csv format logs for each model and metric
- Running the file produces matplotlib graphs of the logs.

This project is an introduciton to machine language translation. The 2 models chosen to train singnify a big milesones in the development of MT neural networks. The Transformer Architecture is proved to be versatile and powerful for not just transalation but also other NLP tasks. The fine tuned model T5 base is also based on the same Transformer architecture. Fine Tuning proved to be a far better approach to traingin LLMs on limited compute resources. 
The T5 model was fine tuned on a very low amount of data and was stll able to translate English text to Finnish text. Which is it was not pre trianed on. It achieved a bleu score of 0.195. 
