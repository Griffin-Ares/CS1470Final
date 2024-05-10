import numpy as np
import librosa
import os
import csv
import pickle
import torch
from tqdm import tqdm
import glob
import torchaudio.transforms as T
from pydub import AudioSegment
import json
import random
import torch
from model import Whisk
import pickle
import numpy as np
from tqdm import tqdm
from models.AlignmentLayer import AlignmentLayer

def apply_spec_augment(spectrogram, freq_mask_param=48, time_mask_param=192, freq_mask_num=2, time_mask_num=2):
    freq_mask = T.FrequencyMasking(freq_mask_param)
    time_mask = T.TimeMasking(time_mask_param)
    
    for _ in range(freq_mask_num):
        spectrogram = freq_mask(spectrogram)
    for _ in range(time_mask_num):
        spectrogram = time_mask(spectrogram)
    
    return spectrogram

def extract_patches_from_spectrogram(y, sr, patch_size=(16, 16), stride=10):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_S = librosa.power_to_db(S, ref=np.max)
    
    log_S_tensor = torch.tensor(log_S).unsqueeze(0).unsqueeze(0)
    log_S_tensor = apply_spec_augment(log_S_tensor)
    log_S = log_S_tensor.squeeze().numpy()
    
    patches = []
    for i in range(0, log_S.shape[1] - patch_size[1] + 1, stride):
        for j in range(0, log_S.shape[0] - patch_size[0] + 1, stride):
            patch = log_S[j:j+patch_size[0], i:i+patch_size[1]]
            flattened_patch = patch.flatten()
            patches.append(flattened_patch)
    return patches


# Use the on a audio file path to get the spectrograms
def process_single_file(file_path, sr=22050):
    # Load file with pydub
    try:
        audio = AudioSegment.from_mp3(file_path)
        # Convert to the appropriate sample rate and channels
        audio = audio.set_frame_rate(sr).set_channels(1)
        # Convert to numpy array
        y = np.array(audio.get_array_of_samples(), dtype=np.float32) / 2**15  # Convert to float32 array
        y = librosa.to_mono(y) if audio.channels > 1 else y

        patches = extract_patches_from_spectrogram(y, sr)
        return file_path, patches
    except Exception as e:
        print(e)
    return None


def preprocess_dev_tsv(file_path):
    # Check if the file exists
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return None

    # Check if the file has the correct extension
    if not file_path.endswith(".tsv"):
        print(f"Invalid file extension: {file_path}")
        return None

    # Initialize an empty list to store the preprocessed data
    preprocessed_data = []

    # Open the TSV file and read its contents
    with open(file_path, "r") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for row in reader:
            # Extract the relevant fields from each row
            index = row["Index"]
            image_key = row["IMAGE_KEY"]
            caption = row["CAPTION_PRED"]
            count_ratings = float(row["COUNT_RATINGS"])
            count_good_ratings = float(row["COUNT_GOOD_RATINGS"])
            avg_user_rating = float(row["ROUNDED_AVG_USER_RATING"])
            model = row["MODEL"]

            # Create a dictionary with the preprocessed data
            preprocessed_row = {
                "index": index,
                "image_key": image_key,
                "caption": caption,
                "count_ratings": count_ratings,
                "count_good_ratings": count_good_ratings,
                "avg_user_rating": avg_user_rating,
                "model": model
            }

            # Append the preprocessed row to the list
            preprocessed_data.append(preprocessed_row)

    return preprocessed_data

def preprocess_meta_tsv(file_path):
    # Check if the file exists
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return None

    # Check if the file has the correct extension
    if not file_path.endswith(".tsv.meta"):
        print(f"Invalid file extension: {file_path}")
        return None

    # Initialize an empty dictionary to store the image_key to originallandingurl mapping
    image_url_map = {}

    # Open the TSV file and read its contents
    with open(file_path, "r") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for row in reader:
            # Extract the image_key and originallandingurl from each row
            image_key = row["IMAGE_KEY"]
            original_landing_url = row["OriginalLandingURL"]

            # Add the image_key and originallandingurl to the dictionary
            image_url_map[image_key] = original_landing_url

    return image_url_map