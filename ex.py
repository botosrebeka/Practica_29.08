import torch
import torchaudio
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def compute_and_plot_stft(audio_path, n_fft=1024, hop_length=512, win_length=1024):
    # Load an audio file
    waveform, sample_rate = torchaudio.load(audio_path, num_frames=480000)

    # Compute STFT
    spectrogram = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(win_length),
        center=False,
        return_complex=True
    )
    # spectrogram = torch.view_as_real(spectrogram)

    # Convert the complex spectrogram to magnitude spectrogram
    magnitude_spectrogram = torch.abs(spectrogram)

    # Display the magnitude spectrogram
    plt.figure(figsize=(10, 6))
    asd = magnitude_spectrogram.log2()[0].numpy()
    plt.imshow(asd, cmap='inferno', aspect='auto', origin='lower')
    plt.colorbar(format="%+2.0f dB")
    plt.title("Magnitude Spectrogram (log scale)")
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency Bins")
    plt.show()

# Usage
audio_path = "5voice.wav"
compute_and_plot_stft(audio_path)


audio_path = "5voicenoise.wav"
compute_and_plot_stft(audio_path)
