import pandas as pd
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
from train_tokenizer import load_tokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Special Tokens
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'

# Token indices
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2

# Configuration
max_len_en = 10
max_len_fr = 10
hidden_size = 512
num_layers = 10
learning_rate = 0.0001
epochs = 10000
batch_size = 256
teacher_forcing_ratio = 0.5

# Preprocessing Functions
def clean_and_lower(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r"[^a-zA-Z0-9\sàâçéèêëîïôûùüÿñæœ'.,!?-]", '', sentence)
    return sentence


def pad_sequences(sequences, max_len, pad_token_idx):
    return [seq[:max_len] + [pad_token_idx] * max(0, max_len - len(seq)) for seq in sequences]

def process_data(file_path, max_len_en, max_len_fr, tokenizer_en, tokenizer_fr):
    df = pd.read_csv(file_path, nrows=1000)
    df['en'] = df['en'].astype(str).apply(clean_and_lower)
    df['fr'] = df['fr'].astype(str).apply(clean_and_lower)

    en_sequences = df['en'].apply(lambda x: tokenizer_en.encode(x, out_type=int)[:max_len_en]).tolist()
    fr_sequences = df['fr'].apply(lambda x: tokenizer_fr.encode(x, out_type=int)[:max_len_fr]).tolist()

    en_padded = pad_sequences(en_sequences, max_len_en, tokenizer_en.piece_to_id(PAD_TOKEN))
    fr_padded = pad_sequences(fr_sequences, max_len_fr, tokenizer_fr.piece_to_id(PAD_TOKEN))

    char_to_index_en = {tokenizer_en.id_to_piece(idx): idx for idx in range(tokenizer_en.get_piece_size())}
    char_to_index_fr = {tokenizer_fr.id_to_piece(idx): idx for idx in range(tokenizer_fr.get_piece_size())}

    return en_padded, fr_padded, char_to_index_en, char_to_index_fr
# Dataset
class TranslationDataset(Dataset):
    def __init__(self, en_data, fr_data):
        self.en_data = torch.tensor(en_data, dtype=torch.long)
        self.fr_data = torch.tensor(fr_data, dtype=torch.long)

    def __len__(self):
        return len(self.en_data)

    def __getitem__(self, index):
        return self.en_data[index], self.fr_data[index]


class EncoderSimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=5):
        super(EncoderSimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn_cell = nn.RNNCell(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        outputs = []
        for t in range(input.size(1)):
            hidden = self.rnn_cell(embedded[:, t, :], hidden)
            outputs.append(hidden)

        return torch.stack(outputs, dim=1), hidden

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size, device=device)


class DecoderSimpleRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=5):
        super(DecoderSimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn_cell = nn.RNNCell(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).squeeze(1)

        hidden = self.rnn_cell(embedded, hidden)

        output = self.out(hidden)
        return output, hidden

# Training Function
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    batch_size = input_tensor.size(0)
    encoder_hidden = encoder.init_hidden(batch_size)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss = 0
    total_correct = 0
    total_chars = 0

    # Encoder forward
    encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)

    # Prepare initial decoder input and hidden state
    decoder_input = torch.tensor([SOS_IDX] * batch_size, device=device).unsqueeze(1)
    decoder_hidden = encoder_hidden

    target_length = target_tensor.size(1)

    for t in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        loss += criterion(decoder_output, target_tensor[:, t])

        # Calculate accuracy for this timestep
        predicted_chars = decoder_output.argmax(1)
        correct = (predicted_chars == target_tensor[:, t]).sum().item()
        total_correct += correct
        total_chars += batch_size

        # Teacher forcing or prediction
        if random.random() < teacher_forcing_ratio:
            decoder_input = target_tensor[:, t].unsqueeze(1)
        else:
            decoder_input = predicted_chars.unsqueeze(1)

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    accuracy = total_correct / total_chars
    return loss.item() / target_length, accuracy

# Translation Function
def translate(encoder, decoder, input_tensor, tokenizer_fr):
    # Place input tensor on the appropriate device

    input_tensor = input_tensor.to(device)
    batch_size = input_tensor.size(0)
    encoder_hidden = encoder.init_hidden(batch_size)

    # Disable gradient tracking during inference
    with torch.no_grad():
        encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)
        decoder_input = torch.tensor([[SOS_IDX]], device=device)
        decoder_hidden = encoder_hidden

        translated_indices = []

        # Generate translation until reaching the max length or EOS token
        for _ in range(max_len_fr):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            top1 = torch.argmax(decoder_output, dim=1).item()

            # Break if EOS is reached
            if top1 == EOS_IDX:
                break

            translated_indices.append(top1)
            decoder_input = torch.tensor([[top1]], device=device)

        # Filter out PAD tokens and decode the remaining indices into text

        translated_indices = [idx for idx in translated_indices if idx != PAD_IDX]
        translated_text = tokenizer_fr.decode(translated_indices)

    return translated_text

# Training Loop
def train_iters(encoder, decoder, dataloader, index_to_char_fr, index_to_char_en):
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # Keep models in training mode
    encoder.train()
    decoder.train()

    for epoch in range(epochs):
        total_loss = 0
        total_accuracy = 0
        total_batches = 0

        for i, (input_tensor, target_tensor) in enumerate(tqdm(dataloader)):
            input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)
            loss, accuracy = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer,
                                   criterion)

            total_loss += loss
            total_accuracy += accuracy
            total_batches += 1

            # Generate example translations without switching to eval() mode
            if (i + 1) % 4 == 0:  # Change frequency for real-time feedback
                print(f"Epoch {epoch + 1}, Step {i + 1}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
                example_input = input_tensor[0].unsqueeze(0)  # Use a single example
                translation = translate(encoder, decoder, example_input, tokenizer_fr)

                # Decode input tensor to its original sentence
                original_sentence = tokenizer_en.decode(example_input[0].cpu().numpy().tolist())


                print(f"Original Sentence (EN): {original_sentence}")
                print(f"Translated Sentence (FR): {translation}")

        average_loss = total_loss / total_batches
        average_accuracy = total_accuracy / total_batches
        print(f"Epoch {epoch + 1}, Average Loss: {average_loss:.4f}, Average Accuracy: {average_accuracy:.4f}")


# Main
if __name__ == "__main__":
    file_path = "../data/en-fr.csv"  # Adjust path

    tokenizer_en = load_tokenizer("en_tokenizer.model")
    tokenizer_fr = load_tokenizer("fr_tokenizer.model")

    en_padded, fr_padded, char_to_index_en, char_to_index_fr = process_data(file_path, max_len_en, max_len_fr, tokenizer_en, tokenizer_fr)

    index_to_char_en = {idx: char for char, idx in char_to_index_en.items()}
    index_to_char_fr = {idx: char for char, idx in char_to_index_fr.items()}

    dataset = TranslationDataset(en_padded, fr_padded)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    encoder = EncoderSimpleRNN(len(char_to_index_en), hidden_size, num_layers).to(device)
    decoder = DecoderSimpleRNN(hidden_size, len(char_to_index_fr), num_layers).to(device)

    train_iters(encoder, decoder, dataloader, tokenizer_en, tokenizer_fr)