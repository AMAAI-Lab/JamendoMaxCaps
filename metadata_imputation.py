import subprocess
import sys
import time
import json
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from retrieve_similar_entries import load_faiss_index, retrieve_top_k, load_features, align_features

TOKEN = 'XXXXXXXXXXXXXXXX'

def install_protobuf():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "protobuf"])

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as e:
    if 'protobuf' in str(e):
        install_protobuf()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    else:
        raise e

def load_local_llm(model_name="meta-llama/Llama-2-7b-hf"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

def enhance_metadata_with_local_llm(captions, metadata_examples, model, tokenizer, device):
    examples = "\n\n".join(
        f"Caption: {caption}\nMetadata: {metadata}"
        for caption, metadata in metadata_examples
    )
    query_caption = captions[-1]
    prompt = (
        "You are an AI language model trained to generate detailed metadata for music tracks based on given captions. "
        "Below are some examples of captions and their corresponding metadata. Use these examples to predict the metadata "
        "for the new caption provided at the end. The metadata should include details such as:\n"
        "- Vocal/Instrumental\n"
        "- Language\n"
        "- Gender\n"
        "- Acoustic/Electric\n"
        "- Speed\n"
        "- Tags (genres, instruments, vartags)\n\n"
        "Examples:\n\n"
        f"{examples}\n\n"
        "New Caption:\n"
        f"{query_caption}\n"
        "Metadata:"
    )

    # Print token count
    inputs = tokenizer(prompt, return_tensors="pt")
    token_count = inputs["input_ids"].shape[1]
    print(f"Token count: {token_count}")

    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    start_time = time.time()
    with torch.cuda.amp.autocast():
        outputs = model.generate(**inputs, max_length=inputs["input_ids"].shape[1] + 500)
    end_time = time.time()
    
    predicted_metadata = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predicted_metadata = predicted_metadata.split("Metadata:")[-1].strip()
    
    inference_time = end_time - start_time
    return predicted_metadata, inference_time

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Metadata imputation.')
    parser.add_argument('--device_index', type=int, required=True, help='CUDA device index (0 to 5)')
    args = parser.parse_args()

    # Set the CUDA device
    device = torch.device(f"cuda:{args.device_index}" if torch.cuda.is_available() else "cpu")

    # Load the corresponding all_keys_{i}.json file
    with open(f'./docs/empty_metadata_ids_{args.device_index}.json', 'r') as f:
        all_keys = json.load(f)

    # Load similar_songs_all.json
    with open('/root/ijcnn/docs/similar_songs_all.json', 'r') as f:
        similar_songs_all = json.load(f)

    # Filter similar_songs_all to include only entries with query_id in all_keys
    similar_songs_all = [entry for entry in similar_songs_all if entry["query_id"] in all_keys]

    # Load metadata
    metadata_features_path = "/data/shared/JamendoMaxCaps/metadata_encoded/encoded_metadata.pt"
    metadata_features, metadata_info = load_features(metadata_features_path)
    metadata_real = {info["id"]: info["musicinfo"] for info in metadata_info}

    # Load model and tokenizer once
    model, tokenizer = load_local_llm()
    model.to(device)

    results_dir = '/root/ijcnn/docs/results'
    os.makedirs(results_dir, exist_ok=True)

    for entry in tqdm(similar_songs_all):
        query_id = entry["query_id"]
        result_file_path = os.path.join(results_dir, f'{query_id}.json')

        # Skip processing if the result file already exists
        if os.path.exists(result_file_path):
            print(f"Skipping query_id {query_id} as result file already exists.")
            continue

        # Log the GPU device and query_id being processed
        print(f"Processing query_id {query_id} on device {device}")

        # Retrieve similar songs and captions from similar_songs_all.json
        similar_songs = entry["similar_songs"]
        example_captions = [song["caption"] for song in similar_songs]
        example_ids = [song["id"] for song in similar_songs]
        example_metadata = [metadata_real[id] for id in example_ids if id in metadata_real]
        print(f"Example captions: {len(example_captions)}")
        example_captions = example_captions[:5]
        example_metadata = example_metadata[:5]
        print(f"Example captions: {len(example_captions)}")

        # Create prompt and predict metadata
        query_caption = entry["query_caption"]
        
        start_time = time.time()
        predicted_metadata, inference_time = enhance_metadata_with_local_llm(
            example_captions + [query_caption],
            list(zip(example_captions, example_metadata)),
            model, tokenizer, device
        )
        end_time = time.time()
        
        total_time = end_time - start_time
        print(f"Time taken for query_id {query_id}: {total_time:.2f} seconds (Inference time: {inference_time:.2f} seconds)")

        result = {
            "id": query_id,
            "predicted_metadata": predicted_metadata,
            "original_musicinfo": metadata_real[query_id],  # Convert Tensor to list
            "caption": query_caption,
            "total_time": total_time,
            "inference_time": inference_time
        }

        # Save result to a JSON file
        with open(result_file_path, 'w') as f:
            json.dump(result, f, indent=4)

    total_end_time = time.time()
    # total_duration = total_end_time - total_start_time
    # print(f"Total time taken: {total_duration:.2f} seconds")

if __name__ == "__main__":
    main()