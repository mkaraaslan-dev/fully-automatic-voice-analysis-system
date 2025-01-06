


from sklearn.metrics import classification_report, confusion_matrix


import numpy as np
import pandas as pd


import torch
import torch.nn as nn
import torchaudio
import torchvision

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


torch.cuda.empty_cache()

seed_value = 42

# PyTorch seed ayarı
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

# Diğer kütüphanelerin seed ayarı (NumPy, random, etc.)
np.random.seed(seed_value)

# Selec Device for Train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DogDatasetDf = pd.read_csv("DogKaggle.csv")

dataNames = DogDatasetDf["data_name"].values
labels = DogDatasetDf['label'].values
# weights_for_classes = torch.Tensor([1 - (len(DogDatasetDf[DogDatasetDf['label'] == x])/len(DogDatasetDf)) for x in ['Bark','Yip','Howl','Bow-wow','Growling','Whimper (dog)']])    

sn.histplot(labels)
plt.savefig("data_distribution.png")
# plt.show()
# label_encoder = preprocessing.LabelEncoder()
# DogDatasetDf["label"] = label_encoder.fit_transform(DogDatasetDf['label'])

x_train, x_test, y_train, y_test = train_test_split(dataNames, labels, test_size=0.2, random_state=42)

print(len(x_train))
config = {
'fixed_sample_rate': 22050,
'hop_length' : 512,
'nfft' : 2048,
'size' : (256,256),
}

# Defining all the transformations
# Wrap it in a function, as sample_rate maybe variable in the data
def preprocess_audio(waveform, sample_rate, config):
    resample_t = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=config['fixed_sample_rate'])
    audio_mono = torch.mean(resample_t(waveform),dim=0, keepdim=True)
    # spectogram_t = torchaudio.transforms.MelSpectrogram(sample_rate, n_fft = config['nfft'], hop_length = config['hop_length'], normalized = True)

    # spectogram_t= torchaudio.transforms.MFCC(
    #     sample_rate=sample_rate,
    #     melkwargs={"n_fft": config['nfft'], "hop_length": config['hop_length']},
    # )
    spectogram_t = torchaudio.transforms.LFCC(sample_rate=sample_rate,
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
    if original_label == 'Bark':
        label = 0
    if original_label == 'Howl':
        label = 1        
    return label



class AudioSetDog(Dataset):
    def __init__(self, data_names,data_labels , preprocess_func, config):
        
        self.labels = data_labels
        self.names = data_names      
        self.preprocess_func = preprocess_func
        self.config = config

    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):

        label = self.labels[index]
        audio_path = f"dog_kag/{label}/{self.names[index]}"
        # get corresponding label from dataframe
        # convert the categorical string data to numerical which works better with our loss function - CrossEntropy
        label = custom_label_encoder(label)
        # get the audio as a Pytorch tensor
        waveform, sample_rate = torchaudio.load(audio_path,normalize = True)
        # pre-process the audio accd to the preprocess function
        audio = self.preprocess_func(waveform, sample_rate, self.config)
        return audio, label
    
train_data = AudioSetDog(x_train,y_train, preprocess_audio, config)
test_data = AudioSetDog(x_test,y_test, preprocess_audio, config)

train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True )



# Move the model to GPU if available; Speeds up training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
def evaluate_svm(predictions, targets, label_encoding, feature_model):
    """Evaluate the SVM model and generate a report."""
    # Map numerical labels back to their original labels
    targets = [label_encoding[label] for label in targets]
    predictions = [label_encoding[label] for label in predictions]

    # Classification report
    report = classification_report(targets, predictions, output_dict=True)
    report2 = classification_report(targets, predictions)
    df = pd.DataFrame(report).transpose()

    # Klasör oluşturma
    save_dir = f"svm_{feature_model}"
    os.makedirs(save_dir, exist_ok=True)

    # Raporu kaydetme
    df.to_csv(os.path.join(save_dir, "classification_report.csv"))

    # Confusion matrix
    matrix = confusion_matrix(targets, predictions)
    plt.clf()
    sn.heatmap(matrix, annot=True, fmt="d", cmap='Blues')
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))

    print("Classification Report:")
    print(report2)
def train_ml_model(model_name, model, train_loader, test_loader, feature_model):
    """Train and evaluate a traditional ML model (SVM, KNN, Random Forest, Naive Bayes)."""
    # Extract features and labels from train_loader
    train_features = []
    train_labels = []
    for inputs, labels in train_loader:
        inputs = inputs.view(inputs.size(0), -1).numpy()  # Flatten inputs
        train_features.extend(inputs)
        train_labels.extend(labels.numpy())

    # Train the ML model
    model.fit(train_features, train_labels)

    # Evaluate on test data
    test_features = []
    test_labels = []
    for inputs, labels in test_loader:
        inputs = inputs.view(inputs.size(0), -1).numpy()  # Flatten inputs
        test_features.extend(inputs)
        test_labels.extend(labels.numpy())

    predictions = model.predict(test_features)
    evaluate_svm(predictions, test_labels, label_encoding, f"{model_name}_{feature_model}")


# Seçmek istediğiniz feature modelini burada ayarlayın
feature_model = "LFCC"  # Mel, MFCC, veya LFCC olabilir

# Makine öğrenmesi modelleri
ml_models = {
    "SVM": make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True, random_state=42)),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "NaiveBayes": GaussianNB()
}

# Tüm modelleri eğit ve değerlendir
for model_name, model in ml_models.items():
    print(f"Training and evaluating {model_name}...")
    train_ml_model(model_name, model, train_dataloader, test_dataloader, feature_model)