import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import re
import random
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
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
hidden_size = 256
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
    fr_sequences = df['fr'].apply(lambda x: tokenizer_fr.encode(x, out_type = int)[:max_len_fr]).tolist()

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


class MyRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyRNNCell, self).__init__()
        self.hidden_size = hidden_size


        self.Wx = nn.Linear(input_size, hidden_size)
        self.Wh = nn.Linear(hidden_size, hidden_size)

    def forward(self, input, hidden):
        new_hidden = torch.tanh(self.Wx(input) + self.Wh(hidden))
        return new_hidden


# Encoder with MyRNNCell
class EncoderCustomRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(EncoderCustomRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)

        self.rnn_cells = nn.ModuleList([MyRNNCell(hidden_size, hidden_size) for _ in range(num_layers)])

    def forward(self, input, hidden):
        embedded = self.embedding(input)  # (batch_size, seq_len, hidden_size)
        outputs = []

        for t in range(input.size(1)):  # Process each timestep
            x_t = embedded[:, t, :]  # Take one token at a time
            for layer in range(self.num_layers):
                hidden[layer] = self.rnn_cells[layer](x_t, hidden[layer])  # Pass through MyRNNCell
                x_t = hidden[layer]  # Update input for next layer
            outputs.append(x_t)

        return torch.stack(outputs, dim=1), hidden

    def init_hidden(self, batch_size):
        return [torch.zeros(batch_size, self.hidden_size, device=device) for _ in range(self.num_layers)]


# Decoder with MyRNNCell
class DecoderCustomRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=1):
        super(DecoderCustomRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(output_size, hidden_size)

        # Initialize MyRNNCell for each layer
        self.rnn_cells = nn.ModuleList([MyRNNCell(hidden_size, hidden_size) for _ in range(num_layers)])

        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).squeeze(1)  # (batch_size, hidden_size)

        x_t = embedded
        for layer in range(self.num_layers):
            hidden[layer] = self.rnn_cells[layer](x_t, hidden[layer])  # Pass through MyRNNCell
            x_t = hidden[layer]  # Update the input for the next layer

        output = self.out(x_t)  # (batch_size, output_size)
        return output, hidden


# Weight Initialization (Xavier Initialization)
def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
        nn.init.xavier_uniform_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)


# Training Function
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    batch_size = input_tensor.size(0)
    encoder_hidden = encoder.init_hidden(batch_size)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss = 0
    total_correct = 0
    total_chars = 0

    # Encoder forward pass
    encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)

    decoder_input = torch.tensor([SOS_IDX] * batch_size, device=device).unsqueeze(1)
    decoder_hidden = encoder_hidden

    target_length = target_tensor.size(1)

    for t in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        loss += criterion(decoder_output, target_tensor[:, t])

        predicted_chars = decoder_output.argmax(1)
        correct = (predicted_chars == target_tensor[:, t]).sum().item()
        total_correct += correct
        total_chars += batch_size

        if random.random() < teacher_forcing_ratio:
            decoder_input = target_tensor[:, t].unsqueeze(1)
        else:
            decoder_input = predicted_chars.unsqueeze(1)

    loss.backward()

    # Gradient clipping
    max_grad_norm = 1.0
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_grad_norm)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_grad_norm)

    encoder_optimizer.step()
    decoder_optimizer.step()

    accuracy = total_correct / total_chars
    return loss.item() / target_length, accuracy


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

            if (i + 1) % 4 == 0:
                print(f"Epoch {epoch + 1}, Step {i + 1}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
                example_input = input_tensor[0].unsqueeze(0)
                translation = translate(encoder, decoder, example_input, tokenizer_fr)

                original_sentence = ''.join([index_to_char_en[idx] for idx in example_input[0].cpu().numpy() if
                                             idx not in [PAD_IDX, SOS_IDX, EOS_IDX]])

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

    encoder = EncoderCustomRNN(len(char_to_index_en), hidden_size, num_layers).to(device)
    decoder = DecoderCustomRNN(hidden_size, len(char_to_index_fr), num_layers).to(device)

    encoder.apply(init_weights)
    decoder.apply(init_weights)

    train_iters(encoder, decoder, dataloader, index_to_char_fr, index_to_char_en)
