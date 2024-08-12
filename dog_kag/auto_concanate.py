
import pandas as pd


file_paths = ['Barklabel.csv','Howllabel.csv']


# Tüm uçuşlara ait verileri birleştirme
all_data = pd.concat([pd.read_csv(file) for file in file_paths])

all_data.to_csv('DogKaggle.csv', index=False)