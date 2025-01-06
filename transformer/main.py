#Importing all neccessary libraries
from transformer_model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR
import warnings
from tqdm import tqdm
import os
from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    """
    This function implements the greedy decoding technique.
    Takes in model, source, source_mask, source tokenizer, 
    target tokenizer , max length, device type as arguments.
    
    returns a tensor
    """
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, 
                   max_len, device, print_msg, global_step, writer,
                   num_examples=2):
    """
    This function runs all the evaluation metrics required for the model.
    In this case BLEU, WER and CER.
    Takes in model, val dataset, source tokenizer, target tokenizer,
    max length, device type, , global step, and number of examples needed to generate 
    for evaluation
    Prints using batch.iterator()function which is passed to the function during call. 
    
    Returns None
    """
    model.eval()
    count = 0
    #Defining variables for source,target and predicted texts
    source_texts = []
    expected = []
    predicted = []

    #Checking console wondow size
    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        #Iterating through the val dataset
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            #Generating prediction
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
            #Save source and target text from dataset.
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            #Save prediction
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            #Appending to list
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")
            #Checking for num_examples limit
            if count == num_examples:
                print_msg('-'*console_width)
                break
    
    if writer:
        # Evaluate the character error rate
        # Compute the char error rate 
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, [expected])
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()
        

def get_all_sentences(ds, lang):
    """
    Gets all sentences from the dataset.
    Takes in dataset and language as inupt
    """
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    """
    This function checks if a tokenizer file already exists
    in the current working directory.
    If not a new tokenizer is trained.
    Takes in config file, dataset and language as arguments
    
    returns tokenizer. 
    """
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    """
    This function loads the dataset unsing the hugging face data loader.
    Takes in the config file, which has the datasource and source and target 
    language names.
    
    It also performs some EDA on the dataset and generates WordCloud images
    for both train adn test dataset.
    
    returns train_dataloader, val_dataloader, source tokenizer and target tokenizer
    
    """
    #Loading the dataset using hugging face dataloader
    ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    # Building tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])
    
    # The dataset only has train split, so we have to create a test split overselves
    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
    #Creating the language pair dataset
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0
    en_text = ""
    fi_text = ""
    for item in ds_raw:
        #Iterating through the dataset and extracting texts for source and target
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        #calculating maximum text length
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
        #Storing all the text into a variable for Wordcloud
        en_text += " " + item['translation'][config['lang_src']]
        fi_text += " " + item['translation'][config['lang_tgt']]
    #plotting the wordcloud for english
    plt.imshow(WordCloud().generate(en_text))
    plt.show()
    #plotting the wordcloud for Finnish
    plt.imshow(WordCloud().generate(fi_text))
    plt.show()       

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    print(f'Vocab size of source sentence: {tokenizer_src.get_vocab_size()}')
    print(f'Vocab size of target sentence: {tokenizer_tgt.get_vocab_size()}')
    #Removing all the text from memory.
    en_text=""
    fi_text=""
    #Creating a iterable dataset for PyTorch to use to train
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    """
    This function uses the method form the file transformer_model.py
    Takes in source vocab length, target vocab length, sequence len and dmodel dimension.
    """
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])
    return model

def train_model(config):
    """
    This function trains the model.
    Takes in config file, checks for device type and trains usign the device.
    Checks to see if model has been trained before, if so loads the pretrained model weights.
    optimiser used is AdamOptimser nad Loss function used is the Croos entropy loss.
    Saves model to specied filepath in the config file at every epoch. 
    """
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    #Assigning device
    device = torch.device(device)

    # Making sure the weights folder exists
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    #Loading the dataset using the get_ds function
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    #Loading the model using the get_model function
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    # Loading Tensorboard
    writer = SummaryWriter(config['experiment_name'])
    #Loading the optimizer function
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # Checking if pretrained model exist, with the specified epoch number.
    initial_epoch = 0
    global_step = 0
    #Checking the preload setting in config file
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    #Loading the model and checking once again if the model exits 
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        
    else:
        print('No model to preload, starting from scratch')
    # loading loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    #Creating a loop to run for num_epochs times
    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        #training the model
        model.train()
        #using tqdm to visualise training
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            #Loading input and mask to device
            encoder_input = batch['encoder_input'].to(device) # (b, seq_len) dimension
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len) dimension
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len) dimension
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len) dimension

            # Running data through the encoder, decoder and projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model) dimension
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model) dimension
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size) dimension

            # Loading the correct output to compare
            label = batch['label'].to(device) # (B, seq_len) dimension

            # Compute the loss
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Logging loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate
            loss.backward()

            # Update weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            #Incrementing global step by 1
            global_step += 1

        # Running validation at the end of every epoch
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

#Load these files first 
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
