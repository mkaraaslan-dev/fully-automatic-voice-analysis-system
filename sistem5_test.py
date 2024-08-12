# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 12:13:27 2022

@author: MAHMUT
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 08:34:33 2022

@author: MAHMUT
"""

import librosa
import torch
import torch.nn as nn
import torchaudio
import torchvision
import matplotlib.pyplot as plt
import numpy as np

from models import *


# file_name = 'sivas_ses_data/1. köpek 5. hafta .wav'
file_name = 'combine7.wav'
audio_data, sr = librosa.load(file_name)




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = AudioResNet152(num_classes=2).to(device)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model.load_state_dict(torch.load("AlexNet_no_weight/model.pth", map_location=device))
model.load_state_dict(torch.load("resnet152_700/final_model.pth", map_location=device))

model.eval()

config = {
'fixed_sample_rate': 22050,
'hop_length' : 128,
'nfft' : 2048,
'size' : (256,256),
}




# nfft = stft için fixed frame size dır. hop lenght bunun yarısı vs..
def plot_spectrogram(specgram, title=None, ylabel="frequency"):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.show(block=False)
    
def preprocess_audio(waveform, sample_rate, config):
    resample_t = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=config['fixed_sample_rate'])
    audio_mono = resample_t(waveform)
    spectogram_t = torchaudio.transforms.MelSpectrogram(sample_rate, n_fft = config['nfft'], hop_length = config['hop_length'], normalized = True)
    sepctogram = spectogram_t(audio_mono)
    plot_spectrogram(sepctogram[0], title="MelSpectrogram - torchaudio", ylabel="mel freq")
    resize_t = torchvision.transforms.Resize(config['size'])
    final_audio = resize_t(sepctogram)
    
    return final_audio


pencere_boyutu = 4096 * 4
# ardışık kesitler arası uzaklık (örnek sayısı cinsinden)
kaydirma_miktari = 4096 * 2
enerji_esik_orani = 0.12
# Birden fazla paramete kullanırken bu parametreleri tek bir sözlük içerisine
# yerleştirip kullanmak daha anlaşılır bir kod yazmamıza yardım edebilir
fs = sr
parametreler = {
    'fs': fs,
    'pencere_boyutu': pencere_boyutu,
    'kaydirma_miktari': kaydirma_miktari,
    'enerji_esik_orani': enerji_esik_orani
}

def dosya_bolutle(ses, parametreler):
    '''
    Enerji eşik değeri kullanarak kaydın sessiz bölgelerinden bölünmesini 
    sağlayacak sınırları tespit eder. Öncelikle kayıt küçük parçalara bölünerek
    her kesit için enerji hesaplanmakta ardından bu enerji değerlerinin eşik
    değeriyle karşılaştırılması sonucu kesitin sesli mi sessiz mi olduğuna karar
    verilmektedir. Bu bilgi bolut_karar_fonk içerisinde 0 ve 1 değerleriyle 
    belirtilmektedir. 0-1 ve 1-0 geçişleri ise kayıtların sınırlarını bulmakta
    kullanılmakta, bunun için bolut_karar_fonk'nun türevi işlenmektedir
    '''

    enerjiler = librosa.feature.rms(ses, frame_length=pencere_boyutu, hop_length=kaydirma_miktari)[0]  
    # plt.title("Root-Mean Square Energy ")
    # plt.plot(enerjiler)
    # listeden numpy-array'e dönüştürme:
    # bu adım diziler üzerinde alttaki işlemleri yapabilmek için gerekli
    enerjiler = np.array(enerjiler)
    # Genlik normalizasyonu
    enerjiler = enerjiler / np.max(enerjiler)

    # Enerji esik oranini kullanarak bolutleme sınırlarının belirlenmesi
    bolut_karar_fonk = np.zeros_like(enerjiler)
    # sessiz kesitler 0, sesli kesitler 1 olarak atanıyor
    bolut_karar_fonk[enerjiler > enerji_esik_orani] = 1
    # Fark fonksiyonu kullanacağımız için başa 0 ekliyoruz
    bolut_karar_fonk = np.insert(bolut_karar_fonk, 0, 0)
    fark_fonksiyonu = np.diff(bolut_karar_fonk)
    # Baslangic endeksleri: 0'dan 1'e geçiş
    baslangic_endeksleri = np.nonzero(fark_fonksiyonu > 0)[
        0] * kaydirma_miktari
    # Bitis endeksleri: 1'den 0'a geçiş
    bitis_endeksleri = np.nonzero(fark_fonksiyonu < 0)[
        0] * kaydirma_miktari
    return (ses, enerjiler, bolut_karar_fonk,
            baslangic_endeksleri, bitis_endeksleri)


(ses, enerjiler, bolut_karar_fonk, baslangic_endeksleri,
bitis_endeksleri) = dosya_bolutle(audio_data, parametreler)
        # Çizimlerin oluşturulması
# print(baslangic_endeksleri)
# print(bitis_endeksleri)
# print(len(baslangic_endeksleri))
# print(len(bitis_endeksleri))


plt.subplot(1, 1, (0+1))
plt.title("audio segment")
plt.plot(ses, label='ses dalgası')
# plt.plot(np.arange(enerjiler.size) * kaydirma_miktari,
#                  enerjiler, 'g', label='enerjiler')
plt.plot(np.arange(bolut_karar_fonk.size) * kaydirma_miktari,
                  bolut_karar_fonk, 'r', label='bölüt karar fonksiyonu')
plt.vlines(baslangic_endeksleri, ymin=-0.5, ymax=0,
                    colors='b', linestyles='solid', label='Bölüt başlangıcı')
plt.vlines(bitis_endeksleri, ymin=-0.5, ymax=0, colors='k',
                    linestyles='dashed', label='Bölüt bitişi')

cnt = 0

# m=nn.Sigmoid()
# m = nn.ReLU()
m = nn.Softmax()


# print(len(baslangic_endeksleri))
# print(len(bitis_endeksleri))

if len(baslangic_endeksleri)-len (bitis_endeksleri)==1:
    bitis_endeksleri=np.append(bitis_endeksleri,len(audio_data))
    

toplam_havlama_sayisi = 0
toplam_havlama_süresi = 0
for i in range(len(bitis_endeksleri)):
    
        start_point = baslangic_endeksleri[i]
        end_point = bitis_endeksleri[i]
        
        audio_segment = torch.FloatTensor(audio_data[start_point:end_point])
        # print(len(audio_segment)/sr)

        # print(len(audio_segment))
        # print(audio_segment.shape)
        audio_spec = preprocess_audio(audio_segment.reshape(1,-1), sr, config)
        inputs = audio_spec.reshape(1,1,256,256)
        inputs = inputs.to(device)
        outputs = model(inputs)
        
        act_out = m(outputs).cpu().detach().numpy()
        
        # y_p = [np.argmax(x) for x in outputs.cpu().detach().numpy()]
        
        # y_p = np.argmax(outputs.cpu().detach().numpy())
        y_p = np.argmax(act_out)
        
        # print("max:",)
                
        # if  
        # print(act_out)
        print("max:",act_out[0][y_p])
        if act_out[0][y_p] >= 0.99:            
            if y_p == [0]:
                havlama_süresi = (end_point - start_point)/config['fixed_sample_rate']
                print("havlama")
                # print("current_havlama süresi: ", havlama_süresi)
                toplam_havlama_sayisi = toplam_havlama_sayisi + 1
                toplam_havlama_süresi = toplam_havlama_süresi + havlama_süresi
                
                # print("havlama süresi")
            elif y_p == 1:
                print("uluma")
        else:
            print("Dış ortam")

    
        # wv.write(f"splitter_test/{cnt}.wav",audio_data[start_point:end_point] , sr, sampwidth=2 )        
        # cnt +=1

# def rms(signal):
    
#     energy = np.sqrt(np.sum(signal**2))
    
#     return energy