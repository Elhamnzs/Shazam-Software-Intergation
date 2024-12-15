import os 

import numpy as np
import librosa 
from scipy.signal import find_peaks

import hashlib
import json





n_fft = 2048
hop_length = n_fft // 4 
sr =  22050


# Define the ROI size
ROI_TIME = 0.5  # in seconds
ROI_FREQ = 100  # in Hz




def spectrogram(y): 

    n_fft = 2048 #  Number of samples in each FFT window. Higher values improve frequency resolution but reduce time resolution
    hop_length = n_fft // 4  # 512, The number of samples between successive FFT frames. Smaller values increase overlap, providing smoother time representation.

    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    spectrogram = np.abs(stft)
    spectrogram_dB = librosa.amplitude_to_db(spectrogram, ref=np.max)

    return spectrogram_dB

def find_peaks_spectrogram(spectrogram_dB): 

    peaks = []
    for t in range(spectrogram_dB.shape[1]):  # Loop over time frames
        
        # Apply the threshold in dB, adjusting the factor if necessary
        # threshold_dB = 0.5 * np.max(spectrogram_dB[:, t])  # threshold in dB
        threshold_dB =  0.2*np.min(spectrogram_dB[:, t])  # threshold in dB
        freq_peaks, _ = find_peaks(spectrogram_dB[:, t], height=threshold_dB)  # Apply threshold to dB values

        for f in freq_peaks:
            frequency = f * sr / n_fft  # Actual frequency in Hz
            time_point = t * hop_length / sr  # Time in seconds
            peaks.append((time_point, frequency))  # Store (time_in_seconds, frequency_bin)


    peaks = np.array(peaks)
    return peaks 


# Function to generate hash
def generate_hash(fa, ta, fk, tk):
    hash_input = f"{fa}-{fk}-{tk-ta}".encode('utf-8')
    return hashlib.sha1(hash_input).hexdigest()

# Select anchor points based on the highest amplitude within a certain region
def select_anchor_points(peaks, spectrogram_dB, num_anchors=5):
    anchor_points = []
    for t in range(0, spectrogram_dB.shape[1], int(ROI_TIME * sr / hop_length)):
        for f in range(0, spectrogram_dB.shape[0], int(ROI_FREQ * n_fft / sr)):

            region_peaks = peaks[
                (peaks[:, 0] >= t * hop_length / sr) & (peaks[:, 0] < (t + int(ROI_TIME * sr / hop_length)) * hop_length / sr) &
                (peaks[:, 1] >= f * sr / n_fft) & (peaks[:, 1] < (f + int(ROI_FREQ * n_fft / sr)) * sr / n_fft)
            ]
            
            if len(region_peaks) > 0:
                max_peak = region_peaks[np.argmax(spectrogram_dB[region_peaks[:, 1].astype(int) * n_fft // sr, region_peaks[:, 0].astype(int) * sr // hop_length])]
                anchor_points.append(max_peak)
    return np.array(anchor_points)


def get_hashes(anchor_points, peaks): 

    # List to store the hashes
    hashes = []

    # Iterate over each anchor point
    for anchor in anchor_points:
        ta, fa = anchor

        # Define the ROI
        roi_peaks = peaks[(peaks[:, 0] >= ta) & (peaks[:, 0] <= ta + ROI_TIME) & (peaks[:, 1] >= fa - ROI_FREQ) & (peaks[:, 1] <= fa + ROI_FREQ)]
        
        # Generate hashes for keypoints in the ROI
        for keypoint in roi_peaks:
            tk, fk = keypoint
            if (tk, fk) != (ta, fa):  # Exclude the anchor point itself
                hash_value = generate_hash(fa, ta, fk, tk)
                hashes.append(((fa, fk, tk - ta), ta, hash_value))

    return hashes


def generate_hash_audio_file(file_path):

    y_full, _ = librosa.load(file_path, sr=sr)
    spectrogram_dB_full = spectrogram(y_full)
    peaks_full= find_peaks_spectrogram(spectrogram_dB_full)
    anchor_points_full = select_anchor_points(peaks_full, spectrogram_dB_full)
    hashes_full = get_hashes(anchor_points_full, peaks_full)

    return list(map(list, hashes_full))  

def generate_hash_audio(y_full):

    spectrogram_dB_full = spectrogram(y_full)
    peaks_full= find_peaks_spectrogram(spectrogram_dB_full)
    anchor_points_full = select_anchor_points(peaks_full, spectrogram_dB_full)
    hashes_full = get_hashes(anchor_points_full, peaks_full)

    return list(map(list, hashes_full))  


def generate_hashes_for_database(database_folder):
    song_hashes = {}
    
    for song_file in os.listdir(database_folder):
        if song_file.endswith('.mp3'):
            audio_file_path = os.path.join(database_folder, song_file)

            print('generating hashes for : ', audio_file_path)
            hashes = generate_hash_audio_file(audio_file_path)

            print('number of hashes generated for : ', audio_file_path, 'is : ', len(hashes))
            print('\n')
            song_hashes[song_file] = hashes

    return song_hashes


# this is the one that gives me good result, the simplest one :) 
def find_best_match(hashes): 

    with open('song_hashes.json', 'r') as f:
        song_hashes = json.load(f)

    # Extract the hash values from the full song hashes 
    hash_values_full = set(h[2] for h in hashes)

    # Find the song with the most matches
    best_match_song = None
    max_matches = 0

    for song, hashes in song_hashes.items():
        hash_values = set(h[2] for h in hashes)
        common_hashes = hash_values_full.intersection(hash_values)
        num_matches = len(common_hashes)

        print('current matches : ', num_matches)
        
        if num_matches > max_matches:
            max_matches = num_matches
            best_match_song = song



    print(f"The song with the most matches is: {best_match_song} with {max_matches} matches")

    return best_match_song, max_matches




if __name__ == '__main__': 


    # creating the database json file
    # TODO: speed up the process with Producer/Consumer multithreading 
    def create_database(): 
        database_folder = 'songs'
        song_hashes = generate_hashes_for_database(database_folder)

        # Save the hashes to a JSON file
        with open('song_hashes.json', 'w') as f:
            json.dump(song_hashes, f)

        print("Hashes generated and saved to song_hashes.json")

    create_database()