import os
import json
import torch
import jsonlines
from collections import defaultdict
from transformers import T5Tokenizer, T5EncoderModel
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator

class MetadataDataset(Dataset):
    def __init__(self, metadata):
        self.metadata = metadata

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        return self.metadata[idx]

def load_metadata(metadata_folder: str):
    """Load metadata from JSONL files."""
    metadata = []
    for file_name in os.listdir(metadata_folder):
        if file_name.endswith(".jsonl"):
            file_path = os.path.join(metadata_folder, file_name)
            with jsonlines.open(file_path) as reader:
                for obj in reader:
                    metadata.append(obj)
    return metadata

def remove_duplicates(metadata):
    print(f'type(metadata): {type(metadata)}')
    """Remove duplicate entries from metadata."""
    unique_metadata = {}
    for item in metadata:
        unique_metadata[item['id']] = item
    return list(unique_metadata.values())

def calculate_max_length(metadata):
    """Calculate the maximum length of the musicinfo field globally."""
    max_length = 0
    for item in metadata:
        musicinfo = item.get('musicinfo', {})
        if musicinfo:
            text_length = len(json.dumps(musicinfo))
            if text_length > max_length:
                max_length = text_length
    return max_length

def encode_musicinfo_batch(batch, text_encoder, tokenizer, device):
    """Encode a batch of musicinfo fields using FLAN-T5."""
    encoded_data = []
    for item in batch:
        musicinfo = item.get('musicinfo', {})
        if not musicinfo:
            print(f"Skipping item {item['id']} as musicinfo is empty.")
            continue  # Skip if the 'musicinfo' field is empty
        text = json.dumps(musicinfo)  # Convert the 'musicinfo' dictionary to a JSON string
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = text_encoder(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).cpu()
        encoded_data.append({
            "id": item["id"],
            "musicinfo": musicinfo,
            "text_info": embedding
        })
    return encoded_data

def save_intermediate_data(encoded_data, output_path, gpu_id, part):
    """Save intermediate encoded data to a .pt file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    intermediate_path = f"{output_path}_gpu{gpu_id}_part{part}.pt"
    torch.save(encoded_data, intermediate_path)

def save_encoded_data(encoded_data, output_path):
    """Save the encoded data to a .pt file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(encoded_data, output_path)

def collate_fn(batch, max_length):
    """Custom collate function to pad sequences to the same length."""
    for item in batch:
        text = json.dumps(item['musicinfo'])
        padding_length = max_length - len(text)
        item['musicinfo'] = text + ' ' * padding_length  # Pad with spaces
    return batch

def main():
    metadata_folder = "/data/shared/JamendoMaxCaps/metadata"  # Update this path to your metadata folder
    output_path = "/data/shared/JamendoMaxCaps/metadata_encoded/encoded_metadata"  # Update this path to your desired output file
    text_encoder_name = "google/flan-t5-base"  # Example text encoder

    # Initialize accelerator
    accelerator = Accelerator()
    device = accelerator.device
    gpu_id = accelerator.process_index

    # Load metadata
    metadata = load_metadata(metadata_folder)
    print(f"Total number of metadata entries before removing duplicates: {len(metadata)}")

    # Remove duplicate entries
    metadata = remove_duplicates(metadata)
    print(f"Total number of metadata entries after removing duplicates: {len(metadata)}")

    # Calculate the maximum length of the musicinfo field globally
    max_length = calculate_max_length(metadata)

    # Create dataset and dataloader
    dataset = MetadataDataset(metadata)
    dataloader = DataLoader(dataset, batch_size=2048, shuffle=False, num_workers=48, collate_fn=lambda x: collate_fn(x, max_length))

    # Load text encoder and tokenizer
    tokenizer = T5Tokenizer.from_pretrained(text_encoder_name)
    text_encoder = T5EncoderModel.from_pretrained(text_encoder_name).to(device)

    # Prepare model and dataloader with accelerator
    text_encoder, dataloader = accelerator.prepare(text_encoder, dataloader)

    # Encode musicinfo
    encoded_data = []
    processed_ids = set()
    skipped_ids = set()
    part = 0
    for batch in tqdm(dataloader, desc="Encoding musicinfo"):
        encoded_batch = encode_musicinfo_batch(batch, text_encoder, tokenizer, device)
        for item in batch:
            if item['id'] in processed_ids or item['id'] in skipped_ids:
                continue
            if any(encoded_item['id'] == item['id'] for encoded_item in encoded_batch):
                processed_ids.add(item['id'])
            else:
                skipped_ids.add(item['id'])
        encoded_data.extend(encoded_batch)
        part += 1
        save_intermediate_data(encoded_data, output_path, gpu_id, part)
        encoded_data = []

    # Combine intermediate results
    if accelerator.is_main_process:
        combined_data = []
        for gpu_id in range(accelerator.num_processes):
            part = 1
            while True:
                intermediate_path = f"{output_path}_gpu{gpu_id}_part{part}.pt"
                if os.path.exists(intermediate_path):
                    data = torch.load(intermediate_path)
                    combined_data.extend(data)
                    part += 1
                else:
                    break
        save_encoded_data(combined_data, f"{output_path}.pt")
        print(f"Encoded data saved to {output_path}.pt")
        print(f"Total processed IDs: {len(processed_ids)}")
        print(f"Total skipped IDs: {len(skipped_ids)}")

if __name__ == "__main__":
    main()
