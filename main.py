import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import random
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import UNet
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# data (title.wav) -> CSV
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


# clasa pentru a crea un custom dataset, care returneaza voice+noise si voice
class CustomSoundDataset(Dataset):

    def __init__(self, voice_annotations_file, noise_annotation_file, voice_dir, noise_dir):
        self.voice_annotations = pd.read_csv(voice_annotations_file)
        self.noise_annotations = pd.read_csv(noise_annotation_file)
        self.voice_dir = voice_dir
        self.noise_dir = noise_dir

    # lungimea datasetului = nr. de samples = 1000
    def __len__(self):
        # return len(self.voice_annotations)
        return 50

    def __getitem__(self, index):
        # seed - hogy mindig azokat hasznald
        voice_state = np.random.RandomState(seed=index)
        noise_state = np.random.RandomState(seed=index)

        # random noise + random voice - shuffle
        self.voice_random_index = voice_state.randint(0, len(self.voice_annotations))
        self.noise_random_index = noise_state.randint(0, len(self.noise_annotations))

        voice_sample_path = self._get_voice_sample_path()
        noise_sample_path = self._get_noise_sample_path()

        voice, vsr = sf.read(voice_sample_path, dtype='float32', always_2d=True)
        noise, nsr = sf.read(noise_sample_path, dtype='float32', always_2d=True)

        voice_norm = self.normalize(voice, -24)
        noise_norm = self.normalize(noise, -30)

        # niste exemple sunt mai lunga decat 480000 sample -> slicing
        # noise = noise/3
        voice_plus_noise = voice_norm[0:480000, :] + noise_norm[0:480000, :]

        return voice_plus_noise.T, voice_norm[0:480000, :].T

    def _get_voice_sample_path(self):
        path = os.path.join(self.voice_dir, self.voice_annotations.iloc[self.voice_random_index, 0])
        return path

    def _get_noise_sample_path(self):
        path = os.path.join(self.noise_dir, self.noise_annotations.iloc[self.noise_random_index, 0])
        return path

    def normalize(self, signal, db):
        r = 10**(db/20.0)
        a = np.sqrt((len(signal) * r**2) / np.sum(signal**2))

        return signal * a


# functie pentru a modifica lungimea
def cut(data, section_length):
    length = len(data) - section_length * len(data) / 10
    _from = random.randrange(0, int(length))
    _to = _from + section_length * len(data) / 10
    _cut = data[int(_from):int(_to)]
    return _cut


# functie pentru a plota voice+noise si voice pe o singura img
def wave_plot(voice_noise, voice):
    plt.subplot(2, 1, 1)
    plt.plot(voice_noise)
    plt.title('Voice + Noise')
    plt.subplot(2, 1, 2)
    plt.plot(voice)
    plt.title('Voice')
    plt.show()


def export_mixed_and_voice(index):
    voice_noise, voice = csd[index]
    vn_string = str(index) + 'voice_noise.wav'
    v_string = str(index) + 'voice.wav'
    sf.write(vn_string, voice_noise, 48000)
    sf.write(v_string, voice, 48000)


def export(name, data):
    name = name + '.wav'
    sf.write(name, data, 48000)


# transform stft - Short Time Fourier Transform - time -> frequency
def transform(signal):
    n_fft = 1024
    hop_length = int(n_fft/2)
    win_length = n_fft
    # print(f"Shape before: {signal.shape}")
    # signal = torch.squeeze(signal)
    signal = torch.squeeze(signal, 1)

    spectrogram = torch.stft(
        signal,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(window_length=win_length).to(device),
        return_complex=True
    )
    # sp = torch.squeeze(spectrogram)
    # plot_spectrogram(sp)
    spectrogram = torch.view_as_real(spectrogram)
    spectrogram = torch.unsqueeze(spectrogram, 1)
    return spectrogram


# functie pentru inverse STFT
def inverse(spectrogram):
    spectrogram = torch.view_as_complex(spectrogram)
    n_fft = 1024
    hop_length = int(n_fft / 2)
    win_length = n_fft

    reconstructed_waveform = torch.istft(
        spectrogram,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(window_length=win_length).to(device),
        return_complex=False,
        length=480000
    )
    return reconstructed_waveform


def plot_spectrogram(data):
    plt.figure(figsize=(10, 6))
    m_sp = torch.abs(data)
    asd = m_sp.log2().numpy()
    plt.imshow(asd, cmap='inferno', aspect='auto', origin='lower')
    plt.colorbar(format="%+2.0f dB")
    plt.title("Magnitude Spectrogram (log scale)")
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency Bins")
    plt.show()


def max_error(a, b):
    return torch.max(torch.abs(a-b))


def compute_magnitude(complex_signal):
    energ = complex_signal[:, :, :, :, 0] ** 2 + complex_signal[:, :, :, :, 1] ** 2
    magnitude = torch.sqrt(energ.clamp_min(1e-17))
    return magnitude


# # Get cpu, gpu or mps device for training.
# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )
device = "cpu"
print(f"Using {device} device")

loss_list = []


def train(model, dataloader, loss_fn, optimizer):
    num_batches = len(dataloader)
    model.train()
    epoch_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        X_trans = transform(X)
        y_trans = transform(y)

        fea = compute_magnitude(X_trans)

        masc = model(fea)

        pred = X_trans * masc.unsqueeze(-1)

        loss = loss_fn(pred, y_trans)
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        epoch_loss += loss

        print(f'Batch: {batch+1}/{num_batches}')

    print(f'Epoch loss: {epoch_loss/num_batches}')


if __name__ == "__main__":
    voice_annotation_file = "voice.csv"
    noise_annotation_file = "noise.csv"
    voice_dir = "E:/Practica/voice"
    noise_dir = "E:/Practica/noise"

    csd = CustomSoundDataset(voice_annotation_file, noise_annotation_file, voice_dir, noise_dir)

    print(f"There are {len(csd)} samples in the dataset.")

    train_dataloader = DataLoader(csd, batch_size=10, shuffle=False)

    for X, y in train_dataloader:
        print(f"Shape of X: {X.shape}")
        print(f"Shape of y: {y.shape}")
        break

    model = UNet.UNetModel().to(device)
    # model = UNet.UNetModel()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.000001)
    epochs = 20

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(model, train_dataloader, loss_fn, optimizer)
        torch.save(model.state_dict(), "model_51.pth")
        print("Saved PyTorch Model State to model_51.pth")
    print("Train done!")

    x = np.linspace(0, len(loss_list), len(loss_list))
    y = loss_list

    plt.plot(x, y)
    plt.show()

    # for X, y in train_dataloader:
    #     print(f"Shape of X: {X.shape}")
    #     print(f"Shape of y: {y.shape}")
    #
    #     x = X[0, :, :]
    #     x = transform(x)
    #
    #     # a = compute_magnitude(x)
    #
    #     # after = inverse(x)
    #     #
    #     # export('after', after)
    #     #
    #     # after = torch.unsqueeze(after, 0)
    #     # print(f"Shape after: {after.shape}")
    #     #
    #     # print(f"Max error: {max_error(before, after)}")
    #     break

