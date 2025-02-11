import librosa
import numpy as np
import os
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def calculate_snr(signal):
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean((signal - np.mean(signal)) ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def calculate_spectral_flatness(y):
    return librosa.feature.spectral_flatness(y=y).mean()

def calculate_zero_crossing_rate(y):
    return librosa.feature.zero_crossing_rate(y).mean()

def calculate_rms_energy(y):
    return librosa.feature.rms(y=y).mean()

def process_file(args):
    file_path, results_dir = args
    try:
        file_id = os.path.basename(file_path).split('.')[0]
        result_file_path = os.path.join(results_dir, f'{file_id}.json')

        y, sr = librosa.load(file_path, sr=None)
        snr = calculate_snr(y)

        if os.path.exists(result_file_path):
            print(f"File {file_path} already exists. Calculating only SNR.")
            with open(result_file_path, 'r') as f:
                result = json.load(f)
            result["snr"] = str(snr)
        else:
            spectral_flatness = calculate_spectral_flatness(y)
            zero_crossing_rate = calculate_zero_crossing_rate(y)
            rms_energy = calculate_rms_energy(y)
            result = {
                "id": int(file_id),
                "spectral_flatness": str(spectral_flatness),
                "zero_crossing_rate": str(zero_crossing_rate),
                "rms_energy": str(rms_energy),
                "snr": str(snr)
            }

        # Save result to a JSON file
        with open(result_file_path, 'w') as f:
            json.dump(result, f, indent=4)

        return result
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Check noisiness of MP3 files.')
    parser.add_argument('--directory', type=str, default='/data/shared/JamendoMaxCaps/jamendo_vbr', help='Directory containing MP3 files')
    parser.add_argument('--results_dir', type=str, default='/root/ijcnn/docs/results_snr', help='Directory to save results')
    args = parser.parse_args()

    # Create results directory if it doesn't exist
    os.makedirs(args.results_dir, exist_ok=True)

    # Get list of all MP3 files in the directory
    mp3_files = [os.path.join(args.directory, f) for f in os.listdir(args.directory) if f.endswith('.mp3')]

    # Use multiprocessing to process files
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(process_file, [(file_path, args.results_dir) for file_path in mp3_files]), total=len(mp3_files), desc="Processing MP3 files"))

    # Filter out None results
    results = [result for result in results if result is not None]

    print(f"Results saved to {args.results_dir}")

if __name__ == "__main__":
    main()
