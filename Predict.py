import torch
from torch.utils.data import DataLoader
import main
import UNet
import soundfile as sf


valid_voice_annotation_file = "voice_valid_dataset.csv"
valid_noise_annotation_file = "noise_valid_dataset.csv"
valid_voice_dir = "E:/Practica/voice"
valid_noise_dir = "E:/Practica/noise"

valid_dataset = main.CustomSoundDataset(valid_voice_annotation_file, valid_noise_annotation_file, valid_voice_dir, valid_noise_dir)

print(f"There are {len(valid_dataset)} samples in the valid dataset.")

valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

model = UNet.UNetModel()
model.load_state_dict(torch.load("model1.pth"))

voice_annotation_file = "voice.csv"
noise_annotation_file = "noise.csv"
voice_dir = "E:/Practica/voice"
noise_dir = "E:/Practica/noise"

csd = main.CustomSoundDataset(voice_annotation_file, noise_annotation_file, voice_dir, noise_dir)

print(f"There are {len(csd)} samples in the dataset.")

train_dataloader = DataLoader(csd, batch_size=4, shuffle=True)


for X, y in train_dataloader:
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")

    model.eval()

    with torch.no_grad():
        asd = torch.squeeze(X)
        asd = asd[1, :]
        sf.write('input.wav', asd, 48000)
        
        X = main.transform(X)
        inp = X
        X = main.compute_magnitude(X)

        masc = model(X)
        masc = torch.unsqueeze(masc, -1)

        pred = inp * masc

        pred = torch.squeeze(pred)
        
        pred = pred[1, :, :, :]
        invers = main.inverse(pred)
        a = 1

        sf.write('out.wav', invers, 48000)

    break
