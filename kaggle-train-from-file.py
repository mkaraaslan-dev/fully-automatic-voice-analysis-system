import os
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import torchvision
from torch.utils.data import Dataset, DataLoader
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from models import *
import pickle

torch.cuda.empty_cache()

# Selec Device for Train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Verinin kaynağını güncelle
train_dir = "dog_dataset/train"
test_dir = "dog_dataset/test"

# Eğitim ve test verilerini ve etiketlerini yükle
def load_data_labels(data_dir):
    data_names = []
    labels = []
    for label_dir in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label_dir)
        if os.path.isdir(label_path):
            for file_name in os.listdir(label_path):
                data_names.append(file_name)
                labels.append(label_dir)
    return data_names, labels

train_dataNames, train_labels = load_data_labels(train_dir)
test_dataNames, test_labels = load_data_labels(test_dir)

config = {
    'fixed_sample_rate': 22050,
    'hop_length' : 512,
    'nfft' : 2048,
    'size' : (64,64),
}

def preprocess_audio(waveform, sample_rate, config):
    resample_t = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=config['fixed_sample_rate'])
    audio_mono = torch.mean(resample_t(waveform),dim=0, keepdim=True)
    # spectogram_t = torchaudio.transforms.MelSpectrogram(config['fixed_sample_rate'], n_fft = config['nfft'], hop_length = config['hop_length'], normalized = True)
    # spectogram_t = torchaudio.transforms.MFCC(
    #     sample_rate=config['fixed_sample_rate'],
    #     melkwargs={"n_fft": config['nfft'], "hop_length": config['hop_length']},
    # )
    spectogram_t = torchaudio.transforms.LFCC(sample_rate=config['fixed_sample_rate'],
                                              speckwargs={"n_fft": config['nfft'], "hop_length": config['hop_length']},
                                              )
    sepctogram = spectogram_t(audio_mono)
    resize_t = torchvision.transforms.Resize(config['size'])
    final_audio = resize_t(sepctogram)
    return final_audio

label_encoding = {
    0: 'Bark',
    1: 'Howl',
}

def custom_label_encoder(original_label):
    return 0 if original_label == 'Bark' else 1 if original_label == 'Howl' else None

def evaluate_model(model, dataset, label_encoding, mode="test"):
    model.eval()
    predictions = []
    targets = []

    # DataLoader kullanarak veri yükleme
    with torch.no_grad():
        data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            targets.extend(labels.numpy())

    # Etiketlerin gerçek değerlerine dönüştürülmesi
    targets = [label_encoding[label] for label in targets]
    predictions = [label_encoding[label] for label in predictions]

    # Sınıflandırma raporu ve karışıklık matrisinin hesaplanması
    report = classification_report(targets, predictions,output_dict=True)
    report2 = classification_report(targets, predictions)
    


    df = pd.DataFrame(report).transpose()
    df.to_csv(f"{mode}_classification_report.csv")

    matrix = confusion_matrix(targets, predictions)
    
    # Görselleştirme ve raporlama
    plt.clf()
    sn.heatmap(matrix, annot=True, fmt="d",cmap='Blues')
    plt.savefig(f"{mode}_confusion_matrix.png")
    
    data = {
        'y_Actual':targets,
        'y_Predicted': predictions
    }

    df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
    pd_confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])

    plt.clf()
    sn.heatmap(pd_confusion_matrix, annot=True,fmt="d", cmap='Blues')
    plt.savefig(f"{mode}_confusion_matrix_with_real_label.png")

    
    print(f"{mode.capitalize()} Classification Report:")
    print(report2)


class AudioSetDog(Dataset):
    def __init__(self, data_names, data_labels, preprocess_func, config, data_dir):
        self.labels = data_labels
        self.names = data_names      
        self.preprocess_func = preprocess_func
        self.config = config
        self.data_dir = data_dir

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        audio_path = os.path.join(self.data_dir, label, self.names[index])
        label = custom_label_encoder(label)
        waveform, sample_rate = torchaudio.load(audio_path, normalize=True)
        audio = self.preprocess_func(waveform, sample_rate, self.config)
        return audio, label

train_data = AudioSetDog(train_dataNames, train_labels, preprocess_audio, config, train_dir)
test_data = AudioSetDog(test_dataNames, test_labels, preprocess_audio, config, test_dir)

train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False)

model = ResNet50(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

train_losses = []
test_losses = []

def train(model, criterion, optimizer, train_loader, test_loader, epochs):
    for it in range(epochs):
        model.train()
        train_loss = []
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        train_loss = np.mean(train_loss)
        model.eval()
        test_loss = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss.append(loss.item())
        
        test_loss = np.mean(test_loss)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        print(f'Epoch {it+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

try:
    train(model, criterion, optimizer, train_dataloader, test_dataloader, 150)
except KeyboardInterrupt:
    print("Training interrupted.")

FILE = f"final_model.pth"
torch.save(model.state_dict(), FILE)

with open('losses.pkl', 'wb') as f:
    pickle.dump(train_losses, f)
    pickle.dump(test_losses, f)

plt.clf()
sn.lineplot(x=range(len(train_losses)), y=train_losses, label='Train_loss')
sn.lineplot(x=range(len(test_losses)), y=test_losses, label='Test_loss')
plt.savefig("loss_figure.png")

# Modeli değerlendirme
evaluate_model(model, test_data, label_encoding, "test")
evaluate_model(model, train_data, label_encoding, "train")
