import pandas as pd
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Special Tokens
SOS_TOKEN = '<bos>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'

# Token indices
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2

# Configuration
max_len_en = 50
max_len_fr = 50
hidden_size = 256
num_layers = 5
learning_rate = 0.0001
epochs = 10000
batch_size = 8
teacher_forcing_ratio = 0.5

# Preprocessing Functions
def clean_and_lower(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r"[^a-zA-Z0-9\sàâçéèêëîïôûùüÿñæœ'.,!?-]", '', sentence)
    return sentence

def char_sequence(sentence):
    return list(sentence)

def add_special_tokens(char_seq):
    return [SOS_TOKEN] + char_seq + [EOS_TOKEN]

def build_vocab(sequences):
    all_chars = set()
    for seq in sequences:
        all_chars.update(seq)
    vocab = {char: i for i, char in enumerate(sorted(list(all_chars)))}
    for token in [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]:
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab

def pad_sequences(sequences, max_len, pad_token_idx):
    return [seq[:max_len] + [pad_token_idx] * max(0, max_len - len(seq)) for seq in sequences]

def process_data(file_path, max_len_en, max_len_fr):
    df = pd.read_csv(file_path, nrows=100)
    df['en'] = df['en'].astype(str).apply(clean_and_lower)
    df['fr'] = df['fr'].astype(str).apply(clean_and_lower)

    en_sequences = df['en'].apply(lambda x: add_special_tokens(char_sequence(x))[:max_len_en]).tolist()
    fr_sequences = df['fr'].apply(lambda x: add_special_tokens(char_sequence(x))[:max_len_fr]).tolist()

    char_to_index_en = build_vocab(en_sequences)
    char_to_index_fr = build_vocab(fr_sequences)

    en_indexed = [[char_to_index_en[char] for char in seq] for seq in en_sequences]
    fr_indexed = [[char_to_index_fr[char] for char in seq] for seq in fr_sequences]

    en_padded = pad_sequences(en_indexed, max_len_en, char_to_index_en[PAD_TOKEN])
    fr_padded = pad_sequences(fr_indexed, max_len_fr, char_to_index_fr[PAD_TOKEN])

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

# Encoder
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=5):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

# Decoder
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=5):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded, hidden)
        output = output.squeeze(1)
        output = self.out(output)

        return output, hidden

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    batch_size = input_tensor.size(0)
    encoder_hidden = encoder.init_hidden(batch_size)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss = 0
    total_correct_top1 = 0
    total_correct_top5 = 0
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

        # Calculate Top-1 and Top-5 Accuracy
        top1_predictions = decoder_output.argmax(1)  # Top-1
        top5_predictions = torch.topk(decoder_output, 5, dim=1).indices  # Top-5

        correct_top1 = (top1_predictions == target_tensor[:, t]).sum().item()
        correct_top5 = sum([1 for i in range(batch_size) if target_tensor[i, t].item() in top5_predictions[i]])

        total_correct_top1 += correct_top1
        total_correct_top5 += correct_top5
        total_chars += batch_size

        # Teacher forcing or prediction
        if random.random() < teacher_forcing_ratio:
            decoder_input = target_tensor[:, t].unsqueeze(1)
        else:
            decoder_input = top1_predictions.unsqueeze(1)

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    # Calculate accuracies for this batch
    accuracy_top1 = total_correct_top1 / total_chars
    accuracy_top5 = total_correct_top5 / total_chars

    return loss.item() / target_length, accuracy_top1, accuracy_top5

# Translation Function
def translate(encoder, decoder, input_tensor, char_to_index_fr, index_to_char_fr):

    with torch.no_grad():  # Disable gradient computation
        input_tensor = input_tensor.to(device)
        batch_size = input_tensor.size(0)
        encoder_hidden = encoder.init_hidden(batch_size)

        encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)

        decoder_input = torch.tensor([SOS_IDX], device=device).unsqueeze(1)
        decoder_hidden = encoder_hidden

        translated_seq = []
        for _ in range(max_len_fr):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            top1 = decoder_output.argmax(1).item()
            if top1 == EOS_IDX:
                break
            translated_seq.append(index_to_char_fr[top1])
            decoder_input = torch.tensor([[top1]], device=device)

        return ''.join(translated_seq)

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
        total_accuracy_top1 = 0
        total_accuracy_top5 = 0
        total_batches = 0

        for i, (input_tensor, target_tensor) in enumerate(tqdm(dataloader)):
            input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)
            loss, accuracy_top1, accuracy_top5 = train(input_tensor, target_tensor, encoder, decoder,
                                                       encoder_optimizer, decoder_optimizer, criterion)

            total_loss += loss
            total_accuracy_top1 += accuracy_top1
            total_accuracy_top5 += accuracy_top5
            total_batches += 1

            # Generate example translations without switching to eval() mode
            if (i + 1) % 10 == 0:  # Change frequency for real-time feedback
                print(f"Epoch {epoch + 1}, Step {i + 1}, Loss: {loss:.4f}, Top-1 Accuracy: {accuracy_top1:.4f}, Top-5 Accuracy: {accuracy_top5:.4f}")
                example_input = input_tensor[0].unsqueeze(0)  # Use a single example
                translation = translate(encoder, decoder, example_input, char_to_index_fr, index_to_char_fr)

                # Decode input tensor to its original sentence
                original_sentence = ''.join([index_to_char_en[idx] for idx in example_input[0].cpu().numpy() if
                                             idx not in [PAD_IDX, SOS_IDX, EOS_IDX]])

                print(f"Original Sentence (EN): {original_sentence}")
                print(f"Translated Sentence (FR): {translation}")

        average_loss = total_loss / total_batches
        average_accuracy_top1 = total_accuracy_top1 / total_batches
        average_accuracy_top5 = total_accuracy_top5 / total_batches

        print(f"Epoch {epoch + 1}, Average Loss: {average_loss:.4f}, Average Top-1 Accuracy: {average_accuracy_top1:.4f}, Average Top-5 Accuracy: {average_accuracy_top5:.4f}")


# Main
if __name__ == "__main__":
    file_path = "../data/en-fr.csv"  # Adjust path
    en_padded, fr_padded, char_to_index_en, char_to_index_fr = process_data(file_path, max_len_en, max_len_fr)

    index_to_char_en = {idx: char for char, idx in char_to_index_en.items()}
    index_to_char_fr = {idx: char for char, idx in char_to_index_fr.items()}

    dataset = TranslationDataset(en_padded, fr_padded)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    encoder = EncoderRNN(len(char_to_index_en), hidden_size, num_layers).to(device)
    decoder = DecoderRNN(hidden_size, len(char_to_index_fr), num_layers).to(device)

    train_iters(encoder, decoder, dataloader, index_to_char_fr, index_to_char_en)
