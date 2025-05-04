import os
os.listdir("/home/padman/Desktop/Shyam/")


dir_name =os.path.join("/home/padman/Desktop/Shyam/" ,"Language Detection Dataset" )
os.listdir(dir_name)

for dirname , _ , files in os.walk(dir_name):
    print(f"{dirname} ,FOLDERS-> {len(_)} , FILES->{len(files)} ")





import matplotlib.pyplot as plt


kannada_path = "/home/padman/Desktop/Shyam/Language Detection Dataset/Kannada"
marathi_path = "/home/padman/Desktop/Shyam/Language Detection Dataset/Marathi"
punjabi_path = "/home/padman/Desktop/Shyam/Language Detection Dataset/Punjabi"
telugu_path = "/home/padman/Desktop/Shyam/Language Detection Dataset/Telugu"
gujarati_path = "/home/padman/Desktop/Shyam/Language Detection Dataset/Gujarati"
malayalam_path = "/home/padman/Desktop/Shyam/Language Detection Dataset/Malayalam"
urdu_path = "/home/padman/Desktop/Shyam/Language Detection Dataset/Urdu"
tamil_path = "/home/padman/Desktop/Shyam/Language Detection Dataset/Tamil"
hindi_path = "/home/padman/Desktop/Shyam/Language Detection Dataset/Hindi"
bengali_path = "/home/padman/Desktop/Shyam/Language Detection Dataset/Bengali"


file_counts = {
    "Kannada": 22208,
    "Marathi": 25379,
    "Punjabi": 26229,
    "Telugu": 23656,
    "Gujarati": 26441,
    "Malayalam": 24044,
    "Urdu": 31960,
    "Tamil": 24196,
    "Hindi": 25462,
    "Bengali": 27258
}

# Pie chart
labels = file_counts.keys()
sizes = file_counts.values()


import os
import librosa
import numpy as np

dir_name = "/home/padman/Desktop/Shyam/Language Detection Dataset/"


print(os.listdir(dir_name))
for folders in os.listdir(dir_name):
    durations = []
    cnt = 0
    for file in os.listdir(os.path.join(dir_name, folders)):
        if cnt>1000:
            break
        if file.endswith(".mp3"):  # add more extensions if needed
            dir_path = os.path.join(dir_name, folders)
            audio_path = os.path.join(dir_path, file)
            # print(audio_path)
            try:
                # print('abcd')
                duration = librosa.get_duration(filename=audio_path)
                # print(duration)
                durations.append(duration)
            except Exception as e:
                print(f"Error processing {file}: {e}")
        cnt+=1

    mean_duration = np.mean(durations)
    median_duration = np.median(durations)
    print(folders)
    print(f"Mean Duration: {mean_duration:.2f} seconds")
    print(f"Median Duration: {median_duration:.2f} seconds")
    print('--------------------------------------------')


import os
import librosa
import numpy as np
import csv


file_counts = {
    "Kannada": 0,
    "Marathi": 1,
    "Punjabi": 2,
    "Telugu": 3,
    "Gujarati": 4,
    "Malayalam": 5,
    "Urdu": 6,
    "Tamil": 7,
    "Hindi": 8,
    "Bengali": 9
}

output_csv = "/home/padman/Desktop/Shyam copy/label/ground_truth.csv"

data_rows = []

for folder in os.listdir(dir_name):
    folder_path = os.path.join(dir_name, folder)
    for file in os.listdir(folder_path):
        if file.endswith(".mp3"):
            audio_path = os.path.join(folder_path, file)
            data_rows.append((audio_path, file_counts[folder]))

with open(output_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["file_path", "language_code"])
    writer.writerows(data_rows)

print(f"CSV written to: {output_csv}")


import pandas as pd

df = pd.read_csv('/home/padman/Desktop/Shyam copy/label/ground_truth.csv')

label_counts = df['language_code'].value_counts().sort_index()
print(label_counts)




import os
import torch
import torchaudio
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
from tqdm import tqdm
import os
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor,
    Trainer,
    TrainingArguments
)

class LanguageIDDataset(Dataset):
    def __init__(self, df, feature_extractor, target_len_sec=5):
        self.df = df
        self.feature_extractor = feature_extractor
        self.target_len = int(16000 * target_len_sec)  # 5 seconds = 80000 samples

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = row["file_path"]
        label = row["language_code"]

        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0)  # convert to mono

        if sr != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)

        total_len = waveform.shape[0]

        if total_len <= self.target_len:
            pad_len = self.target_len - total_len
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
            segments = [waveform]
        else:
            segments = []
            start = 0
            while start < total_len:
                end = min(start + self.target_len, total_len)
                chunk = waveform[start:end]
                if chunk.shape[0] < self.target_len:
                    pad_len = self.target_len - chunk.shape[0]
                    chunk = torch.nn.functional.pad(chunk, (0, pad_len))
                segments.append(chunk)
                start += self.target_len

        # Return first chunk only
        inputs = self.feature_extractor(
            segments[0].numpy(), sampling_rate=16000, return_tensors="pt", padding=False
        )
        input_values = inputs["input_values"].squeeze(0)
        return input_values, torch.tensor(label, dtype=torch.long)
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from transformers import Wav2Vec2FeatureExtractor


# Shuffle and split (80:10:10)
from sklearn.model_selection import train_test_split
import pandas as pd

# Load CSV
df = pd.read_csv("/home/padman/Desktop/Shyam copy/label/ground_truth.csv")  # Update with your actual path

# First split: 80% train, 20% temp (val + test)
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

# Second split: 50% of temp_df goes to val, 50% to test → results in 10% each
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, shuffle=True)


# Initialize Feature Extractor
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")

# Define Datasets
train_dataset = LanguageIDDataset(train_df, feature_extractor)
val_dataset = LanguageIDDataset(val_df, feature_extractor)
test_dataset = LanguageIDDataset(test_df, feature_extractor)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# Visualize one batch
for batch in train_loader:
    input_values, labels = batch
    print("Input shape:", input_values.shape)  # (batch_size, sequence_length)
    print("Labels:", labels)
    break












import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
results_dir = "results"
ckpt_dir = os.path.join(results_dir, "checkpoints")
os.makedirs(results_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)

# Model
class WhisperClassifier(nn.Module):
    def __init__(self, num_labels):
        super(WhisperClassifier, self).__init__()
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-large")
        self.whisper = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")

        self.classifier = nn.Sequential(
            nn.Linear(self.whisper.config.d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_values):
        # Use Whisper's feature extractor (processor)
        features = self.processor(input_values, return_tensors="pt", sampling_rate=16000).input_values.to(device)
        # Pass through Whisper model
        encoder_outputs = self.whisper.encoder(features)
        # Average the encoder outputs across all time steps to obtain a fixed-size representation
        pooled = encoder_outputs.last_hidden_state.mean(dim=1)
        return self.classifier(pooled)

# Hyperparameters
num_classes = df['language_code'].nunique()
model = WhisperClassifier(num_labels=num_classes).to(device)
criterion = nn.CrossEntropyLoss()

# -------------- Logging + Plot Helpers ----------------
def save_plot(data, title, ylabel, filename, labels=None):
    plt.figure()
    for i, series in enumerate(data):
        plt.plot(series, label=labels[i] if labels else None)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    if labels:
        plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, filename))
    plt.close()

def log_to_file(file, epoch, loss, acc, prefix="Train"):
    with open(file, "a") as f:
        f.write(f"{prefix} Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc*100:.2f}%\n")

# ---------------- Feature Extraction ----------------
feature_extract_epochs = 10
optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
for param in model.whisper.parameters():
    param.requires_grad = False

fe_loss, fe_acc = [], []

for epoch in range(feature_extract_epochs):
    model.train()
    epoch_loss, preds_all, labels_all = 0, [], []

    for bcnt, batch in enumerate(train_loader):
        if bcnt >= 100:
            break
        try:
            input_values, labels = [x.to(device) for x in batch]
        except:
            continue

        optimizer.zero_grad()
        outputs = model(input_values)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        preds_all.extend(outputs.argmax(dim=1).cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

        del input_values, labels, outputs, loss
        torch.cuda.empty_cache()
        gc.collect()

    acc = accuracy_score(labels_all, preds_all)
    fe_loss.append(epoch_loss)
    fe_acc.append(acc)

    # Save logs, plots, model
    log_to_file(os.path.join(results_dir, "feature_extraction_log.txt"), epoch, epoch_loss, acc)
    save_plot([fe_loss], "Feature Extraction Loss", "Loss", "feature_extraction_loss.png")
    save_plot([fe_acc], "Feature Extraction Accuracy", "Accuracy", "feature_extraction_accuracy.png")

    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"fe_epoch_{epoch+1}.pt"))

# ---------------- Fine-Tuning ----------------
fine_tune_epochs = 15
for param in model.whisper.parameters():
    param.requires_grad = True
optimizer = optim.Adam(model.parameters(), lr=1e-4)

ft_loss, ft_acc = [], []
val_loss, val_acc = [], []

for epoch in range(fine_tune_epochs):
    model.train()
    epoch_loss, preds_all, labels_all = 0, [], []

    for bcnt, batch in enumerate(train_loader):
        if bcnt >= 200:
            break
        try:
            input_values, labels = [x.to(device) for x in batch]
        except:
            continue

        optimizer.zero_grad()
        outputs = model(input_values)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        preds_all.extend(outputs.argmax(dim=1).cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

        del input_values, labels, outputs, loss
        torch.cuda.empty_cache()
        gc.collect()

    acc = accuracy_score(labels_all, preds_all)
    ft_loss.append(epoch_loss)
    ft_acc.append(acc)

    # Validation
    model.eval()
    val_epoch_loss, val_preds, val_labels = 0, [], []
    with torch.no_grad():
        for vcnt, batch in enumerate(val_loader):
            if vcnt >= 200:
                break
            try:
                input_values, labels = [x.to(device) for x in batch]
            except:
                continue
            outputs = model(input_values)
            loss = criterion(outputs, labels)
            val_epoch_loss += loss.item()
            val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

            del input_values, labels, outputs, loss
            torch.cuda.empty_cache()
            gc.collect()

    val_acc_epoch = accuracy_score(val_labels, val_preds)
    val_loss.append(val_epoch_loss)
    val_acc.append(val_acc_epoch)

    # Save logs, plots, model
    log_to_file(os.path.join(results_dir, "fine_tuning_log.txt"), epoch, epoch_loss, acc, prefix="Train")
    log_to_file(os.path.join(results_dir, "fine_tuning_log.txt"), epoch, val_epoch_loss, val_acc_epoch, prefix="Val")

    save_plot([ft_loss, val_loss], "Fine-Tuning Loss", "Loss", "fine_tuning_loss.png", ["Train", "Val"])
    save_plot([ft_acc, val_acc], "Fine-Tuning Accuracy", "Accuracy", "fine_tuning_accuracy.png", ["Train", "Val"])

    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"ft_epoch_{epoch+1}.pt"))

print("Training complete. Logs, models, and plots saved to 'results/'")