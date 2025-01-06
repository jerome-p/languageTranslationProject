# Machine Language Translation using Deep Learing methods.

This repository contains code for a language translation project.

The repo consists of 3 models 
1. Seq2Seq
2. Transformer
3. Fine-tuned T5 base model.

Running the files
- To run each model, make sure you are in model's folder.
  -  The transformer model is trained by running the main.py file.
  - To train the T5 model, run the language_translation.py file.

The Transformer model follows the implementaion by https://github.com/hkproj/pytorch-transformer
and all the configurations for traingin the model is done via the config file. Edit this file before running the main.py file

Fine tuned model, uses hugging face APIs. All the log files and model files saved by training each of the model will in saved in the model's directory.

Visualisaions.py 
- This file expects a folder with csv format logs for each model and metric
- Running the file produces matplotlib graphs of the logs.
