import os
import shutil
from sklearn.model_selection import train_test_split
import pandas as pd

# Veriyi yükle
DogDatasetDf = pd.read_csv("DogKaggle.csv")
dataNames = DogDatasetDf["data_name"].values
labels = DogDatasetDf['label'].values

# Stratified train-test split ile her sınıftan dengeli örnekleme
train_data, test_data, train_labels, test_labels = train_test_split(
    dataNames, labels, test_size=0.3, random_state=42, stratify=labels
)

# Klasör yapılarını belirle
base_dir = "dog_dataset"  # Ana veri dizini
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# Train ve test klasörlerini ve alt klasörleri oluştur
for label in set(labels):
    os.makedirs(os.path.join(train_dir, label), exist_ok=True)
    os.makedirs(os.path.join(test_dir, label), exist_ok=True)

# Train verilerini klasörlere ayırma
for data_name, label in zip(train_data, train_labels):
    src_path = f"dog_kag/{label}/{data_name}"  # Orijinal dosya yolu
    dest_path = os.path.join(train_dir, label, data_name)  # Yeni konum
    shutil.copy(src_path, dest_path)  # Dosyayı kopyala

# Test verilerini klasörlere ayırma
for data_name, label in zip(test_data, test_labels):
    src_path = f"dog_kag/{label}/{data_name}"  # Orijinal dosya yolu
    dest_path = os.path.join(test_dir, label, data_name)  # Yeni konum
    shutil.copy(src_path, dest_path)  # Dosyayı kopyala

print("Dosyalar dengeli bir şekilde train ve test klasörlerine ayrıldı.")
