
"""



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



This code is broken. The code implementaion is a combination of random fixes from online and suggestions from tools like chatGPT.

None of the results were included in the report. This was the starting point of the project and in the interst of saving time. 
Development on this model was skipped. In the future, I plan to get the code running.

The broken code is still in the repository and not deleted as I was told to include everything done 
in the supervision sessions with the supervisor.

 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


"""

import torch
from torchmetrics import BLEUScore
import seq2seq_2 as s2s

# Define the BLEU evaluation function
bleu = BLEUScore()

def evaluate(encoder, decoder, pairs, max_length=10):
    translations = []
    references = []

    with torch.no_grad():
        for eng_sentence, fi_sentence in pairs:
            # Move input tensors to the correct device
            input_tensor = s2s.train_dataset.tensor_from_sentence(s2s.eng_vocab, s2s.eng_sentence).to(s2s.device)
            input_length = input_tensor.size(0)
            
            # Ensure the encoder_hidden is on the same device
            encoder_hidden = encoder.init_hidden(1).to(s2s.device)  # Move to the correct device

            # Encoder forward pass
            for ei in range(input_length):
                _, encoder_hidden = encoder(input_tensor[ei].unsqueeze(0), encoder_hidden)

            # Prepare the decoder input
            decoder_input = torch.tensor([[s2s.eng_vocab.word2index["SOS"]]]).to(s2s.device)  # SOS token
            decoder_hidden = encoder_hidden

            decoded_words = []
            for di in range(max_length):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                if topi.item() == s2s.fi_vocab.word2index["EOS"]:
                    break
                else:
                    decoded_words.append(s2s.fi_vocab.index2word[topi.item()])

                decoder_input = topi.squeeze().detach()

            translations.append(decoded_words)
            references.append([fi_sentence.split()])  # BLEU expects a list of lists

    bleu_score = bleu(translations, references)
    return bleu_score.item()

# Evaluate the model on the test set and print the BLEU score
bleu_score = evaluate(s2s.encoder, s2s.decoder, s2s.test_pairs)
print(f"BLEU Score: {bleu_score:.4f}")
