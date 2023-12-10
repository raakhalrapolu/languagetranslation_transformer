import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import T5ForConditionalGeneration, AdamW
from tqdm import tqdm

# Paths
tokenized_data_path = 'path_to/your_combined_tokenized_data.pt'
model_save_path = 'path_to/saved_model.pt'

# Load your tokenized data
tokenized_data = torch.load(tokenized_data_path)

# Define a custom dataset
class TranslationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Create a dataset object
dataset = TranslationDataset(tokenized_data)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Define batch size
batch_size = 16  # Adjust as per your GPU memory

# Create dataloaders for training and validation sets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Load the T5 model
model = T5ForConditionalGeneration.from_pretrained('t5-small')  # Or any other variant

# Define an optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# If you're using a GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define number of epochs
num_epochs = 3  # Adjust as per your requirement

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        input_ids, target_ids = batch
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids=input_ids, labels=target_ids)
        loss = outputs.loss

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

    # Validation
    model.eval()
    total_eval_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids, target_ids = batch
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            outputs = model(input_ids=input_ids, labels=target_ids)
            loss = outputs.loss

            total_eval_loss += loss.item()

    print(f"Validation Loss: {total_eval_loss / len(val_loader)}")

# Save the model
torch.save(model.state_dict(), model_save_path)
