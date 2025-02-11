import os
import glob
import torch
import torchaudio  # Import torchaudio
import numpy as np  # Import numpy
from torch.utils.data import DataLoader, Dataset
from typing import Any  # Import Any

class AudioDataset(Dataset):
    def __init__(self, folder_path: str, audio_processor=None, segment_duration: int = 250):
        self.folder_path = folder_path
        self.audio_processor = audio_processor
        self.segment_duration = segment_duration  # Add segment_duration attribute
        if self.audio_processor:
            self.sampling_rate = self.audio_processor.sampling_rate
        # self.sampling_rate = sampling_rate
        self.file_paths = glob.glob(os.path.join(folder_path, "*.mp3"))

    def __len__(self):
        """Return the total number of samples."""
        return len(self.file_paths)

    def __getitem__(self, idx):
        """Load and return a sample from the dataset at the given index.

        Args:
            idx (int): The index of the sample to load.

        Returns:
            dict: A dictionary containing the waveform, file path, and sampling rate.
        """
        file_path = self.file_paths[idx]
        waveform, original_sample_rate = torchaudio.load(file_path)
        if original_sample_rate != self.sampling_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=self.sampling_rate)(waveform)
        target_length = int(self.segment_duration * self.sampling_rate)
        if waveform.shape[1] < target_length:
            # print(f"Padding audio for {file_path}, {waveform.shape[1]} -> {target_length}")
            waveform = torch.nn.functional.pad(waveform, (0, target_length - waveform.shape[1]))
        elif waveform.shape[1] > target_length:
            # print(f"Cutting audio for {file_path}, {waveform.shape[1]} -> {target_length}")
            waveform = waveform[:, :target_length]
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        waveform = self.audio_processor(waveform, sampling_rate=self.sampling_rate)["input_values"][0]  # Process the waveform tensor
        return {
            'waveform': waveform,
            'file_path': file_path,
            'sampling_rate': self.sampling_rate
        }

def create_dataloader(batch_size, num_workers, audio_processor, folder_path: str, segment_duration: int):
    dataset = AudioDataset(folder_path, audio_processor, segment_duration)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    return dataloader