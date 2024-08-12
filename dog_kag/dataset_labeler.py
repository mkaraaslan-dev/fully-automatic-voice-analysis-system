import os
import csv

# Klasör yolu
data_folder_name = 'Howl'

# CSV dosyası adı
csv_dosya_adi = data_folder_name+ 'label.csv'

# Klasördeki dosyaların listesi
dosya_listesi = os.listdir(data_folder_name)

# CSV dosyasını yazma modunda aç
with open(csv_dosya_adi, mode='w', newline='') as dosya:
    yazici = csv.writer(dosya)
    
    # Başlık satırını yaz
    yazici.writerow(['data_name',"label"])
    
    # Dosya listesindeki her dosya için
    for dosya_adi in dosya_listesi:
        # CSV dosyasına dosya ismini yaz
        yazici.writerow([dosya_adi,data_folder_name])
