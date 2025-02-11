import os
import torch
import faiss
import numpy as np
from sklearn.random_projection import SparseRandomProjection
from collections import defaultdict
import argparse

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

def match_features(audio_features, audio_info, metadata_features, metadata_info):
    audio_dict = {os.path.basename(info["file_path"]).split('.')[0]: feature for feature, info in zip(audio_features, audio_info)}
    metadata_dict = {info["id"]: feature for feature, info in zip(metadata_features, metadata_info)}

    common_ids = set(audio_dict.keys()) & set(metadata_dict.keys())
    audio_matched = [audio_dict[id] for id in common_ids]
    metadata_matched = [metadata_dict[id] for id in common_ids]

    print(f"Total number of audio features: {len(audio_features)}")
    print(f"Total number of metadata features: {len(metadata_features)}")
    print(f"Number of matching features: {len(common_ids)}")

    return torch.stack(audio_matched), torch.stack(metadata_matched)

def align_features(audio_features, metadata_features):
    audio_dim = audio_features.shape[-1]
    metadata_dim = metadata_features.shape[-1]

    if audio_dim != metadata_dim:
        rp = SparseRandomProjection(n_components=metadata_dim, random_state=42)
        audio_features_reshaped = audio_features.view(audio_features.size(0), -1)
        audio_features_aligned = rp.fit_transform(audio_features_reshaped)
        audio_features_aligned = torch.tensor(audio_features_aligned, dtype=torch.float32).view(audio_features.size(0), metadata_dim)
    else:
        audio_features_aligned = audio_features

    return audio_features_aligned

def build_faiss_index(features, index_path):
    d = features.shape[-1]
    index = faiss.IndexFlatL2(d)
    features_contiguous = features.view(features.size(0), -1).contiguous()
    index.add(features_contiguous.numpy())
    faiss.write_index(index, index_path)
    return index

def main():
    parser = argparse.ArgumentParser(description="Build FAISS retrieval system")
    parser.add_argument('--weight_audio', type=float, required=True, help='Weight for audio features')
    parser.add_argument('--weight_metadata', type=float, required=True, help='Weight for metadata features')
    args = parser.parse_args()

    audio_features_path = "/data/shared/JamendoMaxCaps/mert_2_test/combined.pt"
    metadata_features_path = "/data/shared/JamendoMaxCaps/metadata_encoded/encoded_metadata.pt"
    index_path = f"./docs/faiss_index_{args.weight_audio}_{args.weight_metadata}.index"

    # Load features
    audio_features, audio_info = load_features(audio_features_path)
    metadata_features, metadata_info = load_features(metadata_features_path)

    # Match features
    audio_matched, metadata_matched = match_features(audio_features, audio_info, metadata_features, metadata_info)

    # Define weights
    weight_audio = args.weight_audio
    weight_metadata = args.weight_metadata

    # Check if weights add up to 1
    if not np.isclose(weight_audio + weight_metadata, 1.0):
        raise ValueError("The weights of audio and metadata must add up to 1.")

    # Align audio features with metadata features if necessary
    if weight_audio != 1.0:
        audio_aligned = align_features(audio_matched, metadata_matched)
        # Normalize embeddings for cosine similarity
        normalized_audio_embeddings = audio_aligned / torch.norm(audio_aligned, dim=-1, keepdim=True)
        normalized_metadata_embeddings = metadata_matched.squeeze(1) / torch.norm(metadata_matched.squeeze(1), dim=-1, keepdim=True)
        # Combine features with weighted sum
        combined_features = weight_audio * normalized_audio_embeddings + weight_metadata * normalized_metadata_embeddings
    else:
        # Reshape normalized_audio_embeddings to [200208, 25*1024] before normalization
        normalized_audio_embeddings = audio_matched.view(audio_matched.size(0), -1)
        # Normalize embeddings for cosine similarity
        normalized_audio_embeddings = normalized_audio_embeddings / torch.norm(normalized_audio_embeddings, dim=-1, keepdim=True)
        combined_features = normalized_audio_embeddings

    # Check dimensions before combining
    print(f"Shape of combined_features: {combined_features.shape}")

    # Build FAISS index
    index = build_faiss_index(combined_features, index_path)
    print(f"FAISS index built and saved to {index_path}")

if __name__ == "__main__":
    main()

