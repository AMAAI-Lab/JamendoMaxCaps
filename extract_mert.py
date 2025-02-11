import argparse
import json
import os
import glob
import numpy as np  # Import numpy
import logging  # Import logging
import time  # Import time for retry mechanism
from typing import Any, Dict, List, Tuple

import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoModel, Wav2Vec2FeatureExtractor
from torch.distributed import init_process_group, destroy_process_group  # Import the necessary functions
from torch.cuda.amp import autocast, GradScaler  # Import for mixed precision
from torch.utils.data import DataLoader  # Import DataLoader
from torch.utils.data._utils.collate import default_collate  # Import default_collate

from dataloader import create_dataloader, AudioDataset  # Import AudioDataset
import yaml  # Import yaml to load the configuration

# Set up logging to console
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

def ddp_setup(rank: int, world_size: int) -> None:
    """Set up the Distributed Data Parallel (DDP) environment.

    Args:
        rank (int): The rank of the current process.
        world_size (int): The total number of processes.
    """
    os.environ["MASTER_ADDR"] = "0.0.0.0"
    os.environ["MASTER_PORT"] = "10056"
    init_process_group(backend="nccl", world_size=world_size, rank=rank)

def download_model_with_retries(model_name: str, retries: int = 5, delay: int = 5):
    """Download the model with retries in case of network issues.

    Args:
        model_name (str): The name of the model to download.
        retries (int): The number of retry attempts.
        delay (int): The delay between retries in seconds.
    """
    for attempt in range(retries):
        try:
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            return model
        except Exception as e:
            logging.error(f"Error during model download: {e}")
            if attempt < retries - 1:
                logging.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise

# def custom_collate_fn(batch):
#     """Custom collate function to pad waveforms to the same length."""
#     max_length = max([item['waveform'].size(0) if isinstance(item['waveform'], torch.Tensor) else item['waveform'].shape[0] for item in batch])
#     for item in batch:
#         if isinstance(item['waveform'], torch.Tensor):
#             if len(item['waveform'].shape) == 1:
#                 item['waveform'] = item['waveform'].unsqueeze(0)  # Ensure waveform is 2D (1, sequence_length)
#             item['waveform'] = torch.nn.functional.pad(item['waveform'], (0, max_length - item['waveform'].size(1)))
#         else:
#             if len(item['waveform'].shape) == 1:
#                 item['waveform'] = np.expand_dims(item['waveform'], axis=0)  # Ensure waveform is 2D (1, sequence_length)
#             item['waveform'] = np.pad(item['waveform'], ((0, 0), (0, max_length - item['waveform'].shape[1])), mode='constant')
#     return default_collate(batch)
def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function. """
    batch_waveforms = torch.stack([torch.from_numpy(item["waveform"]) for item in batch])
    batch_dict = {
        'waveform': batch_waveforms,
        'sampling_rate': [item['sampling_rate'] for item in batch],
        'file_path': [item['file_path'] for item in batch]
    }
    return batch_dict

def create_dataloader(batch_size: int, num_workers: int, audio_processor: Any, folder_path: str, segment_duration: int) -> DataLoader:
    """Create a DataLoader for the dataset.

    Args:
        batch_size (int): The batch size.
        num_workers (int): The number of worker processes.
        audio_processor (Any): The audio processor.
        folder_path (str): The path to the folder containing audio files.
        segment_duration (int): The segment duration for audio files.

    Returns:
        DataLoader: The DataLoader for the dataset.
    """
    dataset = AudioDataset(folder_path, audio_processor, segment_duration)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=custom_collate_fn)

def preprocess_mert_features(gpu_id: int, path_template: str, save_interval: int = 1) -> None:
    """Preprocess audio features using the MERT model.

    Args:
        gpu_id (int): ID of the GPU to use for processing.
        path_template (str): Template for the output file path.
        save_interval (int): Number of batches to process before saving intermediate results.
    """
    folder_path = "/data/shared/JamendoMaxCaps/jamendo_vbr"  # Define the folder path
    logging.info(f"Preprocessing on GPU {gpu_id}")
    segment_duration = 250  # Define the segment duration to match the model's expected input
    audio_processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M", trsut_remote_code=True)
    dataloader = create_dataloader(batch_size=2, num_workers=4, audio_processor=audio_processor, folder_path=folder_path, segment_duration=segment_duration)  # Reduce batch size to 2

    audio_model = download_model_with_retries("m-a-p/MERT-v1-330M").to(gpu_id)
    # scaler = GradScaler()  # Initialize GradScaler for mixed precision

    audio_model.eval()
    batch_counter = 0
    source: List[torch.Tensor] = []
    track_info: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Processing on GPU {gpu_id}"):
            waveforms = batch['waveform'].squeeze(1).to(gpu_id)
            max_length = max([waveform.size(0) for waveform in waveforms])
            padded_waveforms = torch.stack([torch.nn.functional.pad(waveform, (0, max_length - waveform.size(0))) for waveform in waveforms])
            hidden_states = audio_model(padded_waveforms, output_hidden_states=True).hidden_states
            audio_features = torch.stack([h.detach() for h in hidden_states], dim=1)
            audio_features = audio_features.mean(dim=-2)
            source.append(audio_features.cpu())
            track_info.extend([{'file_path': file_path, 'sampling_rate': sampling_rate}
                               for file_path, sampling_rate in zip(batch['file_path'], batch['sampling_rate'])])

            batch_counter += 1
            if batch_counter % save_interval == 0:
                save_intermediate_results(gpu_id, path_template, source, track_info, batch_counter // save_interval)
                source = []
                track_info = []

    # Save any remaining data
    if source:
        save_intermediate_results(gpu_id, path_template, source, track_info, (batch_counter // save_interval) + 1)

def save_intermediate_results(gpu_id: int, path_template: str, source: List[torch.Tensor], track_info: List[Dict[str, Any]], part: int) -> None:
    """Save intermediate results to disk.

    Args:
        gpu_id (int): ID of the GPU to use for processing.
        path_template (str): Template for the output file path.
        source (List[torch.Tensor]): List of audio features.
        track_info (List[Dict[str, Any]]): List of track information.
        part (int): Part number for the intermediate results.
    """
    source_tensor = torch.cat(source, dim=0)
    save_dict = {"source": source_tensor, "track_info": track_info}
    torch.save(save_dict, path_template.format(gpu_id, part))

def main(rank: int, world_size: int, path_template: str) -> None:
    """Main function to run the preprocessing on a single GPU.

    Args:
        rank (int): The rank of the current process.
        world_size (int): The total number of processes.
        path_template (str): Template for the output file path.
    """
    try:
        ddp_setup(rank, world_size)
        preprocess_mert_features(rank, path_template)
    except Exception as e:
        logging.error(f"Exception in process {rank}: {e}")
    finally:
        destroy_process_group()

def sanity_check(folder_path: str, processed_files: List[str]) -> None:
    """Sanity check to ensure all files have been processed.

    Args:
        folder_path (str): The path to the folder containing MP3 files.
        processed_files (List[str]): List of processed file paths.
    """
    all_files = set(glob.glob(os.path.join(folder_path, "*.mp3")))
    processed_files_set = set(processed_files)
    missing_files = all_files - processed_files_set

    if missing_files:
        logging.info(f"Total missing files: {len(missing_files)}")
        # print(f"Missing files: {missing_files}")
    else:
        logging.info("All files have been processed.")

if __name__ == "__main__":
    output_folder = f"/data/shared/JamendoMaxCaps/mert"
    path_template = f"{output_folder}/partition_{{0}}_part_{{1}}.pt"
    combined_file_path = f"{output_folder}/combined.pt"
    folder_path = "/data/shared/JamendoMaxCaps/jamendo_vbr"  # Define the folder path

    os.makedirs(output_folder, exist_ok=True)

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, path_template), nprocs=world_size)

    # After all processes finish, combine the preprocessed features
    combined_source: List[torch.Tensor] = []
    combined_track_info: List[Dict[str, Any]] = []
    processed_files: List[str] = []

    for gpu_id in range(world_size):
        part = 1
        while True:
            file_path = path_template.format(gpu_id, part)
            if os.path.exists(file_path):
                # print(f"Loading {file_path}")  # Commented out
                data = torch.load(file_path)
                combined_source.append(data["source"])
                combined_track_info.extend(data["track_info"])
                processed_files.extend([info['file_path'] for info in data["track_info"]])
                part += 1
            else:
                break

    if combined_source:
        combined_source_tensor = torch.cat(combined_source, dim=0)
        save_dict = {
            "source": combined_source_tensor,
            "track_info": combined_track_info,
        }
        torch.save(save_dict, combined_file_path)
        logging.info(f"Combined preprocessed features saved to {combined_file_path}")
    else:
        logging.info("No features were combined. The combined_source list is empty.")

    # Perform sanity check
    sanity_check(folder_path, processed_files)