import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import random
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def toCSV(directory, name):

    # get a list of all files in the directory
    all_files = os.listdir(directory)

    # filter this list to include only .wav files
    wav_files = [f for f in all_files if f.endswith('.wav')]

    num_files = len(wav_files)
    train_index = int(num_files*0.9)
    valid_index = int(num_files*0.95)

    train_files = wav_files[:train_index]
    valid_files = wav_files[train_index:valid_index]
    test_files = wav_files[valid_index:]

    # create a DataFrame from the list of wav files
    train_df = pd.DataFrame([os.path.join(f) for f in train_files], columns=['name'])
    valid_df = pd.DataFrame([os.path.join(f) for f in valid_files], columns=['name'])
    test_df = pd.DataFrame([os.path.join(f) for f in test_files], columns=['name'])

    # write the DataFrame to a CSV file
    train_csv = name + '_train_dataset.csv'
    valid_csv = name + '_valid_dataset.csv'
    test_csv = name + '_test_dataset.csv'

    train_df.to_csv(train_csv, index=False)
    valid_df.to_csv(valid_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print('CSV files created successfully for ' + name + '!')

# toCSV("E:/Practica/voice", 'voice')
# toCSV("E:/Practica/noise", 'noise')

# voice_train_data = pd.read_csv('voice_test_dataset.csv')
# noise_train_data = pd.read_csv('noise_test_dataset.csv')

class CustomSoundDataset(Dataset):

    def __init__(self, voice_annotations_file, noise_annotation_file, voice_dir, noise_dir):
        self.voice_annotations = pd.read_csv(voice_annotations_file)
        self.noise_annotations = pd.read_csv(noise_annotation_file)
        self.voice_dir = voice_dir
        self.noise_dir = noise_dir

    def __len__(self):
        # return len(self.voice_annotations)
        return 1000

    def __getitem__(self, index):
        voice_state = np.random.RandomState(seed=index)
        noise_state = np.random.RandomState(seed=index)

        self.voice_random_index = voice_state.randint(0, len(self.voice_annotations))
        self.noise_random_index = noise_state.randint(0, len(self.noise_annotations))

        voice_sample_path = self._get_voice_sample_path(index)
        noise_sample_path = self._get_noise_sample_path(index)

        voice, vsr = sf.read(voice_sample_path, dtype='float32', always_2d=True)
        noise, nsr = sf.read(noise_sample_path, dtype='float32', always_2d=True)
        voice_plus_noise = voice[0:480000, :] + noise[0:480000, :]
        a = 1
        return voice_plus_noise.T, voice[0:480000, :].T

    def _get_voice_sample_path(self, index):
        path = os.path.join(self.voice_dir, self.voice_annotations.iloc[self.voice_random_index, 0])
        return path

    def _get_noise_sample_path(self, index):
        path = os.path.join(self.noise_dir, self.noise_annotations.iloc[self.noise_random_index, 0])
        return path

def cut(data, section_lenght):
    lenght = len(data) - section_lenght * len(data)/10
    _from = random.randrange(0, int(lenght))
    _to = _from + section_lenght * len(data)/10
    cutted = data[int(_from):int(_to)]
    return cutted

def wave_plot(voicenoise, voice):
    plt.subplot(2, 1, 1)
    plt.plot(voicenoise)
    plt.title('Voice + Noise')
    plt.subplot(2, 1, 2)
    plt.plot(voice)
    plt.title('Voice')
    plt.show()

def export_mixed_and_voice(index):
    voicenoise, voice = csd[index]
    vnstring = str(index) + 'voicenoise.wav'
    vstring = str(index) + 'voice.wav'
    sf.write(vnstring, voicenoise, 48000)
    sf.write(vstring, voice, 48000)

def export(name, data):
    name = name + '.wav'
    sf.write(name, data, 48000)

def transform(x):
    n_fft = 512
    hop_length = int(n_fft/2)
    win_length = n_fft
    print(f"Shape before: {x.shape}")
    x = torch.squeeze(x)

    spectogram = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(window_length=win_length),
        return_complex=True
    )
    plot_spectogram(spectogram)
    spectogram = torch.view_as_real(spectogram)
    return spectogram

def inverse(x):
    x = torch.view_as_complex(x)
    n_fft = 512
    hop_length = int(n_fft / 2)
    win_length = n_fft

    reconstructed_waveform = torch.istft(
    x,
    n_fft=n_fft,
    hop_length=hop_length,
    win_length=win_length,
    window=torch.hann_window(window_length=win_length),
    return_complex=False
    )
    return reconstructed_waveform

if __name__ == "__main__":
    voice_annotation_file = "voice_train_dataset.csv"
    noise_annotation_file = "noise_train_dataset.csv"
    voice_dir = "E:/Practica/voice"
    noise_dir = "E:/Practica/noise"

    csd = CustomSoundDataset(voice_annotation_file, noise_annotation_file, voice_dir, noise_dir)

    print(f"There are {len(csd)} samples in the dataset.")

    # plot un exemplu din dataset
    # voicenoise, voice = csd[250]
    # wave_plot(voicenoise.T, voice.T)

    # asd, _ = csd[1]
    # export('proba', cut(asd, 3))

    # wave_plot(voicenoise, voice)

train_dataloader = DataLoader(csd, batch_size=32, shuffle=True)

def plot_spectogram(data):
    plt.figure(figsize=(10, 6))
    m_sp = torch.abs(data)
    asd = m_sp.log2().numpy()
    plt.imshow(asd, cmap='inferno', aspect='auto', origin='lower')
    plt.colorbar(format="%+2.0f dB")
    plt.title("Magnitude Spectrogram (log scale)")
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency Bins")
    plt.show()

def eroare(a,b):
    return torch.max(torch.abs(a-b))

def compute_magnitude(complex_signal):
    a = complex_signal
    magnitude = torch.sqrt(complex_signal[:, :, 0] ** 2 + complex_signal[:, :, 1] ** 2)
    return magnitude

for X, y in train_dataloader:
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")

    before = X[0, :, :]
    export('inainte', torch.squeeze(before, 0))

    x = X[0, :, :]
    x = transform(x)

    a = compute_magnitude(x)


    after = inverse(x)

    export('dupa', after)

    after = torch.unsqueeze(after, 0)
    print(f"Shape after: {after.shape}")

    print(f"Eroare maxima: {eroare(before, after)}")


    # y = transform(y[0, :, :])
    # plot_spectogram(y)
    break

#
# plt.plot(a)
# plt.show()


# for X, y in csd:
#     print(f"Shape of X: {X.shape}")
#     print(f"Shape of y: {y.shape}")
#     break



# for i in range(5):
#     m, v = csd[i]
#     wave_plot(m, v)

    # plt.title(len(cut(voicenoise, 5)))
    # plt.plot(voicenoise)
    # plt.plot(cut(voicenoise, 5))
    # plt.show()

    # plt.plot(voicenoise)
    # plt.plot(voice)
    # plt.show()


