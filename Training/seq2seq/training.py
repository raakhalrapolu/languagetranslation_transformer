import random
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from tokenizers import ByteLevelBPETokenizer
from sklearn.model_selection import train_test_split


# Data loading and preprocessing functions
def load_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = file.readlines()
            print(f"Loaded data from {file_path}")
        return data
    except Exception as e:
        print(f"An error occurred while loading data from {file_path}: {e}")
        return []

def preprocess_data(czech_sentences, english_sentences):
    data = []
    for ces, eng in zip(czech_sentences, english_sentences):
        input_text = f"translate Czech to English: {ces.strip()}"
        target_text = eng.strip()
        data.append({"input_text": input_text, "target_text": target_text})
    return data

ces_sentences = load_data('/home/pandu3011/data/wmt22-csen/train.ces')
eng_sentences = load_data('/home/pandu3011/data/wmt22-csen/train.eng')

preprocessed_data = preprocess_data(ces_sentences, eng_sentences)



# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer(
    vocab_file='/home/pandu3011/data/tokeniser-pt/vocab.json', 
    merges_file='/home/pandu3011/data/tokeniser-pt/merges.txt'
)

# Customize training with your dataset paths and tokenizer parameters
tokenizer.train(files=['/home/pandu3011/data/czech_data.txt', '/home/pandu3011/data/english_data.txt'],
                vocab_size=30_000,  # Size of the vocabulary
                min_frequency=2,   # Minimum frequency for a token to be included
                special_tokens=[
                    "<pad>",
                    "<s>",
                    "</s>",
                    "<unk>",
                    "<mask>",
                ])

# Save the trained tokenizer
tokenizer.save_model('/home/pandu3011/data/tokeniser.pt')




def batch_tokenize(data, batch_size=32):
    batched_data = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        input_texts = [item['input_text'] for item in batch]
        target_texts = [item['target_text'] for item in batch]
        tokenized_inputs = tokenizer.encode_batch(input_texts)
        tokenized_targets = tokenizer.encode_batch(target_texts)
        input_ids = torch.tensor([item.ids for item in tokenized_inputs])
        target_ids = torch.tensor([item.ids for item in tokenized_targets])
        batched_data.append((input_ids, target_ids))
    return batched_data


INPUT_DIM = 30_000  # Approximate to your vocab_size
OUTPUT_DIM = 30_000  # Approximate to your vocab_size
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
ENC_DROPOUT = 0.2
DEC_DROPOUT = 0.2
PAD_IDX = tokenizer.token_to_id("<pad>")

# DataLoader
class TranslationDataset(Dataset):
    def __init__(self, preprocessed_data):
        self.data = preprocessed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(torch.sum(energy, dim=2))
        return torch.softmax(attention.squeeze(2), dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        a = self.attention(hidden, encoder_outputs)
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        assert (output == hidden).all()
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        return prediction, hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)
        input = trg[0,:]
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        return outputs

dataset = TranslationDataset(batch_tokenize(preprocessed_data))
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(dataset, batch_size=32)  # Adjust as needed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, HID_DIM, ENC_DROPOUT)
attn = Attention(HID_DIM, HID_DIM)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, HID_DIM, DEC_DROPOUT, attn)


model = Seq2Seq(enc, dec, device).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = optim.Adam(model.parameters())

# Splitting the data into training and validation sets
train_data, valid_data = train_test_split(preprocessed_data, test_size=0.2, random_state=42)

# Creating datasets for training and validation
train_dataset = TranslationDataset(batch_tokenize(train_data))
valid_dataset = TranslationDataset(batch_tokenize(valid_data))

# DataLoader for the training set
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# DataLoader for the validation set
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# Training Loop
def train(model, loader, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for _, (src, trg) in enumerate(tqdm(loader)):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(loader)

# Validation Loop
def evaluate(model, loader, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for _, (src, trg) in enumerate(tqdm(loader)):
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, 0)  # turn off teacher forcing
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(loader)

# Training and Evaluation
N_EPOCHS = 10
CLIP = 1

for epoch in range(N_EPOCHS):
    train_loss = train(model, train_loader, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_loader, criterion)

    print(f'Epoch: {epoch+1:02}')
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f}')

    # Save the model every 3 epochs
    if (epoch + 1) % 3 == 0:
        torch.save(model.state_dict(), f'model_epoch{epoch+1}.pt')
