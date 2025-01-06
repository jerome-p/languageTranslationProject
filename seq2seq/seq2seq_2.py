
"""



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



This code is broken. The code implementaion is a combination of random fixes from online and suggestions from tools like chatGPT.

None of the results were included in the report. This was the starting point of the project and in the interst of saving time. 
Development on this model was skipped. In the future, I plan to get the code running.

The broken code is still in the repository and not deleted as I was told to include everything done 
in the supervision sessions with the supervisor.

 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


"""


import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.text import BLEUScore


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path for saving the model
MODEL_PATH = "seq2seq2_checkpoints/seq2seq_model.pth"

# Load OPUS Books dataset
dataset = load_dataset("opus_books", "en-fi", split='train')

# Keep 90% for training, 10% for validation
train_ds_size = int(0.9 * len(dataset))
val_ds_size = len(dataset) - train_ds_size
train_data, test_data = random_split(dataset, [train_ds_size, val_ds_size])

# Tokenizer
def tokenize_pair(pair):
    return pair['translation']['en'], pair['translation']['fi']

train_pairs = [tokenize_pair(ex) for ex in train_data]
test_pairs = [tokenize_pair(ex) for ex in test_data]

# Vocabulary preparation
class Vocabulary:
    def __init__(self):
        self.word2index = {"SOS": 0, "EOS": 1}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.word2count = {}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.word2count[word] = 1
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Build vocabularies for English and Finnish
eng_vocab = Vocabulary()
fi_vocab = Vocabulary()

for eng, fi in train_pairs:
    eng_vocab.add_sentence(eng)
    fi_vocab.add_sentence(fi)

# Dataset preparation
class TranslationDataset(Dataset):
    def __init__(self, pairs, eng_vocab, fi_vocab, max_length=10):
        self.pairs = pairs
        self.eng_vocab = eng_vocab
        self.fi_vocab = fi_vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        eng_sentence, fi_sentence = self.pairs[idx]
        eng_tensor = self.tensor_from_sentence(self.eng_vocab, eng_sentence)
        fi_tensor = self.tensor_from_sentence(self.fi_vocab, fi_sentence)
        return eng_tensor, fi_tensor

    def tensor_from_sentence(self, vocab, sentence):
        indexes = [vocab.word2index[word] for word in sentence.split(' ') if word in vocab.word2index]
        indexes.append(vocab.word2index["EOS"])  # Append EOS token
        return torch.tensor(indexes, dtype=torch.long)

# DataLoader
train_dataset = TranslationDataset(train_pairs, eng_vocab, fi_vocab)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=lambda x: x)

# Seq2Seq Model (Encoder and Decoder)
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        input = input.to(device)  # Move input to device
        embedded = self.embedding(input).view(1, 1, -1)  # Embedding lookup
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = torch.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

# Function to check if a model has been trained and save the model if the loss improves
def save_model_if_improved(epoch_loss, best_loss, model, optimizer, epoch):
    if epoch_loss < best_loss:
        print(f"Loss improved, saving model at epoch {epoch + 1} with loss {epoch_loss:.4f}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, MODEL_PATH)
        return epoch_loss
    return best_loss

# Function to load the model if it has been saved before
def load_model_if_exists(model, optimizer):
    if os.path.exists(MODEL_PATH):
        print("Loading previously trained model...")
        checkpoint = torch.load(MODEL_PATH)
        
        # Load the checkpoint, ignoring size mismatch errors for incompatible layers
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Model loaded. Last training loss: {loss:.4f}")
        return epoch, loss
    else:
        print("No pre-trained model found. Starting training from scratch.")
        return 0, float('inf')  # No prior training, so set best loss to infinity


# Initialize model, optimizer, and load previous model if exists
hidden_size = 256
encoder = EncoderRNN(eng_vocab.n_words, hidden_size).to(device)
decoder = DecoderRNN(hidden_size, fi_vocab.n_words).to(device)
encoder_optimizer = optim.SGD(encoder.parameters(), lr=0.01)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=0.01)

# Load model if it exists
epoch_start, best_loss = load_model_if_exists(encoder, encoder_optimizer)

# Loss and optimizer
criterion = nn.CrossEntropyLoss().to(device)

# Initialize TensorBoard SummaryWriter
writer = SummaryWriter(log_dir='./runs/seq2seq_training_2')

# Training step
def train_step(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=10):
    input_tensor = input_tensor.to(device)
    target_tensor = target_tensor.to(device)

    if input_tensor.dim() == 1:
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

    if target_tensor.dim() == 1:
        target_tensor = target_tensor.unsqueeze(0)  # Add batch dimension

    batch_size = input_tensor.size(0)
    encoder_hidden = encoder.init_hidden(batch_size).to(device)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss = 0

    # Encoder forward pass
    for ei in range(input_tensor.size(1)):
        encoder_output, encoder_hidden = encoder(input_tensor[:, ei], encoder_hidden)

    # Decoder forward pass
    decoder_input = torch.tensor([[eng_vocab.word2index["SOS"]]]).to(device)  # SOS token
    decoder_hidden = encoder_hidden

    for di in range(target_tensor.size(1)):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        loss += criterion(decoder_output, target_tensor[:, di])  # Calculate loss using target tensor's di-th word
        decoder_input = target_tensor[:, di].unsqueeze(0)  # Teacher forcing

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_tensor.size(1)

# Training loop with TensorBoard logging
num_epochs = 30
for epoch in range(epoch_start, num_epochs):
    total_loss = 0
    encoder.train()
    decoder.train()
    
    with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
        for batch_idx, batch in enumerate(pbar):
            input_tensors, target_tensors = zip(*batch)
            input_tensors = nn.utils.rnn.pad_sequence(input_tensors, padding_value=eng_vocab.word2index['EOS']).to(device)
            target_tensors = nn.utils.rnn.pad_sequence(target_tensors, padding_value=fi_vocab.word2index['EOS']).to(device)

            input_tensors = input_tensors.T.to(device)
            target_tensors = target_tensors.T.to(device)

            batch_loss = 0
            for input_tensor, target_tensor in zip(input_tensors, target_tensors):
                loss = train_step(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
                batch_loss += loss

            total_loss += batch_loss
            avg_loss = total_loss / (batch_idx + 1)

            pbar.set_postfix(loss=avg_loss)

    epoch_loss = total_loss / len(train_loader)
    
    # Save the model if loss improves
    best_loss = save_model_if_improved(epoch_loss, best_loss, encoder, encoder_optimizer, epoch)

    # Log the loss to TensorBoard
    writer.add_scalar('train loss', loss, epoch)

    writer.add_scalar('Loss/epoch', epoch_loss, epoch)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Close the TensorBoard writer
writer.close()

