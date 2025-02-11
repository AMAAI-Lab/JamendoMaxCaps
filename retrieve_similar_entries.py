import torch
import faiss
import numpy as np
import os
import json
import argparse
from sklearn.random_projection import SparseRandomProjection
from tqdm import tqdm

def load_faiss_index(index_path):
    index = faiss.read_index(index_path)
    return index

def retrieve_top_k(index, query_embeddings, k=10):
    query_embeddings = query_embeddings.reshape(query_embeddings.size(0), -1).contiguous()
    # print(f"Query embeddings shape: {query_embeddings.shape}")
    if query_embeddings.is_cuda:
        query_embeddings = query_embeddings.cpu().numpy()
    else:
        query_embeddings = query_embeddings.numpy()
    distances, indices = index.search(query_embeddings, k)
    return distances, indices

def load_features(file_path):
    data = torch.load(file_path)
    if isinstance(data, dict):
        return data["source"], data["track_info"]
    elif isinstance(data, list):
        features = [item["text_info"] for item in data]
        info = [{"id": item["id"], "musicinfo": item["musicinfo"]} for item in data]
        return torch.stack(features), info
    else:
        raise TypeError("Unsupported data format")

def align_features(audio_features, metadata_features):
    # print(f"Audio features shape: {audio_features.shape}, Metadata features shape: {metadata_features.shape}")
    audio_dim = audio_features.shape[-1]
    metadata_dim = metadata_features.shape[-1]

    if audio_dim != metadata_dim:
        rp = SparseRandomProjection(n_components=metadata_dim, random_state=42)
        audio_features_reshaped = audio_features.view(audio_features.size(0), -1)
        if audio_features_reshaped.is_cuda:
            audio_features_reshaped = audio_features_reshaped.cpu().numpy()
        else:
            audio_features_reshaped = audio_features_reshaped.numpy()
        audio_features_aligned = rp.fit_transform(audio_features_reshaped)
        audio_features_aligned = torch.tensor(audio_features_aligned, dtype=torch.float32).view(audio_features.size(0), metadata_dim).to(audio_features.device)
    else:
        audio_features_aligned = audio_features

    return audio_features_aligned

def process_batch(query_ids, index, audio_dict, metadata_dict, captions, audio_info, weight_audio, weight_metadata, k, device):
    audio_queries = []
    metadata_queries = []
    for query_id in query_ids:
        if query_id not in audio_dict or query_id not in metadata_dict:
            print(f"ID {query_id} not found in audio or metadata features.")
            continue

        audio_query = audio_dict[query_id].unsqueeze(0).to(device)
        metadata_query = metadata_dict[query_id].to(device)
        audio_queries.append(audio_query)
        metadata_queries.append(metadata_query)
        # print(f'audio query: {audio_query.shape}, metadata query: {metadata_query.shape}')

    if not audio_queries or not metadata_queries:
        return []

    audio_queries = torch.cat(audio_queries, dim=0)
    metadata_queries = torch.cat(metadata_queries, dim=0)
    # print(f'audio queries: {len(audio_queries)}, metadata queries: {len(metadata_queries)}')
    

    # Align audio features with metadata features if necessary
    if weight_audio != 1.0:
        audio_aligned = align_features(audio_queries, metadata_queries)
        # Normalize embeddings for cosine similarity
        normalized_audio_embeddings = audio_aligned / torch.norm(audio_aligned, dim=-1, keepdim=True)
        normalized_metadata_embeddings = metadata_queries / torch.norm(metadata_queries, dim=-1, keepdim=True)
        # Combine features with weighted sum
        combined_queries = weight_audio * normalized_audio_embeddings + weight_metadata * normalized_metadata_embeddings
    else:
        # Reshape normalized_audio_embeddings to [batch_size, 25*1024] before normalization
        normalized_audio_embeddings = audio_queries.view(audio_queries.size(0), -1)
        # Normalize embeddings for cosine similarity
        normalized_audio_embeddings = normalized_audio_embeddings / torch.norm(normalized_audio_embeddings, dim=-1, keepdim=True)
        combined_queries = normalized_audio_embeddings

    # print(f"Combined queries shape: {combined_queries.shape}")
    distances, indices = retrieve_top_k(index, combined_queries, k)

    results = []
    for i, query_id in enumerate(query_ids):
        top_k_ids = [audio_info[idx]["file_path"].split('/')[-1].split('.')[0] for idx in indices[i]]
        result = {
            "query_id": query_id,
            "query_caption": captions.get(query_id, ""),  # Use empty string if key is not found
            "similar_songs": [{"id": top_k_id, "caption": captions.get(top_k_id, "")} for top_k_id in top_k_ids]  # Use empty string if key is not found
        }
        results.append(result)

    return results

def process_all_ids(audio_features_path, metadata_features_path, captions_path, output_path, weight_audio, weight_metadata, k=10, batch_size=32):
    # Load FAISS index
    index_path = f"./docs/faiss_index_{weight_audio}_{weight_metadata}.index"
    index = load_faiss_index(index_path)
    
    if not np.isclose(weight_audio + weight_metadata, 1.0):
        raise ValueError("The weights of audio and metadata must add up to 1.")
    # Load features
    audio_features, audio_info = load_features(audio_features_path)
    metadata_features, metadata_info = load_features(metadata_features_path)

    # Load captions
    captions = {}
    with open(captions_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            if 'file_path' in item and 'caption' in item:
                file_path = item['file_path']
                music_id = os.path.basename(file_path).split('.')[0]
                captions[music_id] = item['caption']

    # Match features
    audio_dict = {os.path.basename(info["file_path"]).split('.')[0]: feature for feature, info in zip(audio_features, audio_info)}
    metadata_dict = {info["id"]: feature for feature, info in zip(metadata_features, metadata_info)}

    # Process each ID in batches
    query_ids = list(audio_dict.keys())
    results = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for i in tqdm(range(0, len(query_ids), batch_size)):
        batch_query_ids = query_ids[i:i + batch_size]
        batch_results = process_batch(batch_query_ids, index, audio_dict, metadata_dict, captions, audio_info, weight_audio, weight_metadata, k, device)
        results.extend(batch_results)

    # Save results to a JSON file
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Retrieve similar entries by ID")
    parser.add_argument('--id', type=str, required=True, help='ID to search for')
    parser.add_argument('--weight_audio', type=float, required=True, help='Weight for audio features')
    parser.add_argument('--weight_metadata', type=float, required=True, help='Weight for metadata features')
    args = parser.parse_args()

    index_path = f"./docs/faiss_index_{args.weight_audio}_{args.weight_metadata}.index"
    audio_features_path = "/data/shared/JamendoMaxCaps/mert_2_test/combined.pt"
    metadata_features_path = "/data/shared/JamendoMaxCaps/metadata_encoded/encoded_metadata.pt"
    query_id = args.id  # Get the ID from command line arguments

    # Load FAISS index
    index = load_faiss_index(index_path)

    # Load features
    audio_features, audio_info = load_features(audio_features_path)
    metadata_features, metadata_info = load_features(metadata_features_path)

    # Match features
    audio_dict = {os.path.basename(info["file_path"]).split('.')[0]: feature for feature, info in zip(audio_features, audio_info)}
    metadata_dict = {info["id"]: feature for feature, info in zip(metadata_features, metadata_info)}

    if query_id not in audio_dict or query_id not in metadata_dict:
        print(f"ID {query_id} not found in audio or metadata features.")
        return

    audio_query = audio_dict[query_id].unsqueeze(0)
    metadata_query = metadata_dict[query_id].unsqueeze(0)

    # Define weights
    weight_audio = args.weight_audio
    weight_metadata = args.weight_metadata

    # Check if weights add up to 1
    if not np.isclose(weight_audio + weight_metadata, 1.0):
        raise ValueError("The weights of audio and metadata must add up to 1.")

    # Align audio features with metadata features if necessary
    if weight_audio != 1.0:
        audio_aligned = align_features(audio_query, metadata_query)
        # Normalize embeddings for cosine similarity
        normalized_audio_embeddings = audio_aligned / torch.norm(audio_aligned, dim=-1, keepdim=True)
        normalized_metadata_embeddings = metadata_query / torch.norm(metadata_query, dim=-1, keepdim=True)
        # Combine features with weighted sum
        combined_query = weight_audio * normalized_audio_embeddings + weight_metadata * normalized_metadata_embeddings
        print(f"normalized audio embeddings: {normalized_audio_embeddings.shape}, normalized metadata embeddings: {normalized_metadata_embeddings.shape}, combined query: {combined_query.shape}")
    else:
        # Reshape normalized_audio_embeddings to [1, 25*1024] before normalization
        normalized_audio_embeddings = audio_query.view(audio_query.size(0), -1)
        # Normalize embeddings for cosine similarity
        normalized_audio_embeddings = normalized_audio_embeddings / torch.norm(normalized_audio_embeddings, dim=-1, keepdim=True)
        combined_query = normalized_audio_embeddings

    # Retrieve top-k similar entries
    k = 10
    distances, indices = retrieve_top_k(index, combined_query, k)

    # Get the original file IDs for the top-k similar entries
    top_k_ids = [audio_info[idx]["file_path"].split('/')[-1].split('.')[0] for idx in indices[0]]

    print(f"Top-{k} similar entries for query ID {query_id}:")
    for i in range(k):
        print(f"ID: {top_k_ids[i]}, Distance: {distances[0][i]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve similar entries for all IDs")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    process_all_ids(
        config['audio_features_path'],
        config['metadata_features_path'],
        config['captions_path'],
        config['output_path'],
        config['weight_audio'],
        config['weight_metadata'],
        config['k'],
        batch_size=1024  # Add batch_size parameter
    )
