import torch
from torch.utils.data import DataLoader
import main
import UNet
import soundfile as sf
import numpy as np

model = UNet.UNetModel()
model.load_state_dict(torch.load("model.pth"))

voice_annotation_file = "voice.csv"
noise_annotation_file = "noise.csv"
voice_dir = "E:/Practica/voice"
noise_dir = "E:/Practica/noise"

csd = main.CustomSoundDataset(voice_annotation_file, noise_annotation_file, voice_dir, noise_dir)

print(f"There are {len(csd)} samples in the dataset.")

train_dataloader = DataLoader(csd, batch_size=2, shuffle=True)

for X, y in train_dataloader:
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")

    model.eval()

    with torch.no_grad():
        asd = torch.squeeze(X, 1)
        asd = asd[0, :]
        sf.write('predict_input.wav', asd, 48000)

        X = main.transform(X)
        inp = X
        X = main.compute_magnitude(X)

        masc = model(X)
        masc = torch.unsqueeze(masc, -1)

        pred = inp * masc

        zgomot = inp - pred
        zgomot = torch.squeeze(zgomot, 1)
        zgomot = zgomot[0, :, :, :]
        inv = main.inverse(zgomot)
        sf.write('zgomot.wav', inv, 48000)

        pred = torch.squeeze(pred, 1)

        pred = pred[0, :, :, :]
        invers = main.inverse(pred)

        sf.write('predict_output.wav', invers, 48000)

    break


