


from sklearn.metrics import classification_report, confusion_matrix


import numpy as np
import pandas as pd


import torch
import torch.nn as nn
import torchaudio
import torchvision

from torch.utils.data import Dataset
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import preprocessing

from models import *
import pickle

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
model = AudioResNet152(num_classes=2).to(device)


# Loss and optimizer
# criterion = nn.CrossEntropyLoss(weight = weights_for_classes.to(device))
criterion = nn.CrossEntropyLoss()

# optimizer = torch.optim.Adam(model.parameters())
optimizer = torch.optim.SGD(model.parameters(), lr = 0.0001, momentum = 0.9)
# A function to encapsulate the training loop
train_losses = []
test_losses = []
def train(model, criterion, optimizer, train_loader, test_loader, epochs):

    for it in range(epochs):
        model.train()
        train_loss = []
        for inputs, targets in train_loader:
            # move data to GPU
            inputs, targets = inputs.to(device), targets.to(device)
            # print(inputs)
            # zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # Backward and optimize
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        # Get train loss and test loss
        train_loss = np.mean(train_loss)
        # Early Stopping to prevent overfitting on train data
        # if (train_loss < 0.03):
        #     print("Stopping Early...")
        #     break
        model.eval()
        test_loss = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss.append(loss.item())
        test_loss = np.mean(test_loss)
        # Save losses
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        print(f'Epoch {it+1}/{epochs}, Train Loss: {train_loss:.4f}, \tTest Loss: {test_loss:.4f}')
        if (it+1)%350 == 0:
            FILE = f"model_{it+1}.pth"
            torch.save(model.state_dict(), FILE )
            with open('losses.pkl', 'wb') as f:
                pickle.dump(train_losses, f)
                pickle.dump(test_losses, f) 

                print("Veriler başarıyla kaydedildi.")
            
            torch.save({'epoch': it+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss}, 
	        'model.pth')

try:
    train(model, criterion, optimizer, train_dataloader,test_dataloader, 700)
except KeyboardInterrupt:
    print("KeyboardInterrupt")


FILE = f"final_model.pth"
torch.save(model.state_dict(), FILE )
    
with open('losses.pkl', 'wb') as f:
    pickle.dump(train_losses, f)
    pickle.dump(test_losses, f) 

    print("Losslar  başarıyla kaydedildi.")


# model.load_state_dict(torch.load('model_1100.pth', map_location=device))
# print(test_losses,train_losses)
            
# checkpoint = torch.load('model.pth')
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']

# print("epoch: ",epoch)
# print("loss: ",loss)
# print("")



# Verileri dosyadan yükleme
# with open('losses.pkl', 'rb') as f:
#     train_losses = pickle.load(f)
#     test_losses = pickle.load(f)

plt.clf()
sn.lineplot(x = range(len(train_losses)), y = train_losses, label = 'Train_loss')
sn.lineplot(x = range(len(test_losses)), y = test_losses, label = 'Test_loss')
plt.savefig("loss_figure.png")


# # Sınıflandırma raporu ve karışıklık matrisi
evaluate_model(model, test_data, label_encoding, "test")
evaluate_model(model, train_data, label_encoding, "train")


