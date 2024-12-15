import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.io.wavfile import read as wav_read
from scipy.ndimage import maximum_filter
import os
import pickle
import numpy as np
import threading
import librosa
import gc
import psutil
import threading
from scipy.io.wavfile import write
import sounddevice as sd
import pickle


def load_audio(file_path):
    """
    Load an audio file (MP3 or WAV) and return the audio data and sample rate.

    Args:
        file_path: Path to the audio file.

    Returns:
        data: Audio signal as a NumPy array.
        sample_rate: Sample rate of the audio signal.
    """
    # Load the audio file
    data, sample_rate = librosa.load(file_path, sr=None)  # sr=None preserves the original sample rate

    # Convert stereo to mono if needed
    if len(data.shape) > 1:
        data = librosa.to_mono(data)

    return data, sample_rate


def generate_spectrogram(file_path, nperseg=2048, noverlap=1024, nfft=2048):
    """
    Generate a spectrogram using fixed parameters optimized for Shazam-like performance.

    Args:
        file_path: Path to the audio file (WAV format).
        nperseg: Length of each segment for STFT (window size).
        noverlap: Overlap between segments.
        nfft: Number of FFT points.

    Returns:
        f: Frequencies.
        t: Time.
        Zxx: STFT result (complex values).
    """
    # Load the audio file
    data, sample_rate = load_audio(file_path)

    # Convert to mono if stereo
    if len(data.shape) > 1:
        data = data.mean(axis=1)

    # Perform STFT
    f, t, Zxx = stft(data, fs=sample_rate, nperseg=nperseg, noverlap=noverlap, nfft=nfft)

    # Plot the spectrogram (optional for debugging)

    return f, t, Zxx


def extract_key_points_shazam(Zxx, threshold=0.5, neighborhood_size=(20, 20)):
    """
    Extract sparse key points (peaks) from the spectrogram using Shazam-like methods.

    Args:
        Zxx: STFT spectrogram (complex values).
        threshold: Fraction of the max intensity to filter peaks.
        neighborhood_size: Tuple defining the frequency and time region for peak detection.

    Returns:
        key_points: List of (frequency, time) indices for the key points.
    """
    # Compute magnitude of the spectrogram
    magnitude = np.abs(Zxx)

    # Local maxima detection
    local_max = maximum_filter(magnitude, size=neighborhood_size)
    peaks = (magnitude == local_max) & (magnitude > threshold * magnitude.max())

    # Extract key points as (frequency, time) indices
    key_points = np.argwhere(peaks)

    # Debugging: Print key points
    print(f"Extracted {len(key_points)} key points: {key_points[:10]}")  # Print first 10 key points for debugging

    return key_points


def choose_anchor_points(
    key_points,
    freq_bin_size=250,       # Adjust frequency bin size
    time_bin_size=5,         # Adjust time bin size
    max_freq_distance=2000,  # Adjust maximum frequency range for anchor point pairing
    min_time_distance=1,     # Keep a minimal time gap
    max_time_distance=10     # Adjust maximum time difference for anchor point pairing
):
    """
    Choose anchor points using Shazam-like fixed grid spacing and ROI.

    Args:
        key_points: List of (frequency, time) key points.
        freq_bin_size: Frequency bin size for fixed grid spacing.
        time_bin_size: Time bin size for fixed grid spacing.
        max_freq_distance: Maximum frequency range for anchor point pairing.
        min_time_distance: Minimum time difference for anchor point pairing.
        max_time_distance: Maximum time difference for anchor point pairing.

    Returns:
        anchor_points: List of anchor point pairs ((f_a, t_a), (f_b, t_b)).
    """
    # Step 1: Sort key points by time
    key_points = sorted(key_points, key=lambda x: x[1])  # Sort by time

    # Step 2: Group key points into fixed grid cells
    grid = {}
    for f, t in key_points:
        freq_bin = int(f // freq_bin_size)  # Determine frequency bin
        time_bin = int(t // time_bin_size)  # Determine time bin
        cell_id = (freq_bin, time_bin)

        # Store the first key point in each grid cell
        if cell_id not in grid:
            grid[cell_id] = (f, t)  # Keep the first point in the cell

    # Step 3: Extract reference points from the grid
    reference_points = list(grid.values())

    # Step 4: Create anchor points
    anchor_points = []
    for f_a, t_a in reference_points:
        for f_b, t_b in key_points:
            if t_b <= t_a:  # Skip points that are not in the future
                continue
            time_diff = t_b - t_a
            freq_diff = abs(f_b - f_a)
            if min_time_distance <= time_diff <= max_time_distance and freq_diff <= max_freq_distance:
                # Add valid anchor point pair
                anchor_points.append(((f_a, t_a), (f_b, t_b)))

    # Debugging: Print anchor points
    print(f"Generated {len(anchor_points)} anchor points: {anchor_points[:10]}")  # Print first 10 anchor points for debugging

    return anchor_points


def hash_anchor_points(anchor_points):
    """
    Generate hashes from anchor points for Shazam-like music recognition.

    Args:
        anchor_points: List of anchor point pairs ((f_a, t_a), (f_b, t_b)).

    Returns:
        hashes: List of hashes, where each hash is a tuple (f_a, f_b, delta_t, t_a).
    """
    hashes = []
    for (f_a, t_a), (f_b, t_b) in anchor_points:
        delta_t = t_b - t_a  # Time difference
        hash_value = (f_a, f_b, delta_t)  # Create a hash from frequencies and time difference
        hashes.append((hash_value, t_a))  # Store hash with the reference time
    # Debugging: Print generated hashes
    print(f"Generated {len(hashes)} hashes: {hashes[:10]}")  # Print first 10 hashes for debugging
    return hashes


def store_hashes(hashes, song_id, database_file="hash_database.pkl"):
    """
    Store hashes in a hash database.

    Args:
        hashes: List of hashes (hash_value, reference_time).
        song_id: Identifier for the song (e.g., song name or ID).
        database_file: Path to the database file.

    Returns:
        None
    """
    try:
        # Load existing database
        with open(database_file, "rb") as db:
            hash_database = pickle.load(db)
    except FileNotFoundError:
        # Initialize an empty database if the file doesn't exist
        hash_database = {}

    # Add hashes for the current song
    if song_id not in hash_database:
        hash_database[song_id] = []
    hash_database[song_id].extend(hashes)

    # Save updated database
    with open(database_file, "wb") as db:
        pickle.dump(hash_database, db)

    print(f"Hashes stored for song: {song_id}")
    # Debugging: Print stored hashes
    print(f"Stored hashes for {song_id}: {hash_database[song_id][:10]}")  # Print first 10 hashes for debugging


def match_hashes(live_hashes, database_file="hash_database.pkl"):
    """
    Match live recording hashes against the hash database using frequency pairs and delta_t comparison.

    Args:
        live_hashes: List of hashes (hash_value, reference_time) for the live recording.
        database_file: Path to the hash database.

    Returns:
        best_match: The song ID with the highest number of matching hashes.
    """
    try:
        # Load the hash database
        with open(database_file, "rb") as db:
            hash_database = pickle.load(db)
    except FileNotFoundError:
        print("Error: Hash database not found.")
        return None

    # Count matches for each song
    match_counts = {}
    for live_hash, _ in live_hashes:
        live_f_a, live_f_b, live_delta_t = live_hash  # Use delta_t
        for song_id, song_hashes in hash_database.items():
            for song_hash, _ in song_hashes:
                song_f_a, song_f_b, song_delta_t = song_hash  # Use delta_t
                # Compare the frequency pairs and delta_t
                if live_f_a == song_f_a and live_f_b == song_f_b and live_delta_t == song_delta_t:
                    match_counts[song_id] = match_counts.get(song_id, 0) + 1

    # Find the song with the highest match count
    if match_counts:
        best_match = max(match_counts, key=match_counts.get)
        print(f"Best Match: {best_match} with {match_counts[best_match]} matches")
        return best_match
    else:
        print("No Match Found")
        return None


def process_and_store_song_in_thread(song_file, database_file="hash_database.pkl"):
    def process_and_store():
        try:
            # Extract song data
            song_id = os.path.basename(song_file)
            print(f"Processing: {song_id}")

            # Step 1: Generate spectrogram
            f, t, Zxx = generate_spectrogram(song_file, nperseg=2048, noverlap=1024)
            print(f"Spectrogram Shape for {song_id}: {Zxx.shape}")

            # Step 2: Extract key points
            key_points = extract_key_points_shazam(Zxx, threshold=0.3, neighborhood_size=(30, 30))
            print(f"Key Points for {song_id}: {len(key_points)}")

            # Step 3: Generate anchor points
            anchor_points = choose_anchor_points(
                key_points,
                freq_bin_size=500,
                time_bin_size=10,
                max_freq_distance=5000,
                min_time_distance=1,
                max_time_distance=15
            )
            print(f"Anchor Points for {song_id}: {len(anchor_points)}")

            # Step 4: Generate hashes
            hashes = hash_anchor_points(anchor_points)
            print(f"Hashes Generated for {song_id}: {len(hashes)}")

            # Step 5: Store in database (thread-safe)
            # with database_lock:
            try:
                # Load existing database
                with open(database_file, "rb") as db:
                    hash_database = pickle.load(db)
            except FileNotFoundError:
                hash_database = {}

                # Add new song data
                hash_database[song_id] = hashes

                # Save updated database
                with open(database_file, "wb") as db:
                    pickle.dump(hash_database, db)

            print(f"Database updated successfully for {song_id}.")
            # Debugging: Print current database state
            print(f"Current database state: {list(hash_database.keys())}")

        except Exception as e:
            print(f"Error processing {song_file}: {e}")

    # without threading 
    process_and_store()

    # # Run the processing task in a separate thread
    # thread = threading.Thread(target=process_and_store)
    # thread.start()
    # return thread



def inspect_hash_database(database_file="hash_database.pkl"):
    """
    Load and inspect the hash database.

    Args:
        database_file: Path to the hash database file.

    Returns:
        None
    """
    try:
        # Load the database
        with open(database_file, "rb") as db:
            hash_database = pickle.load(db)

        # Print a summary
        print(f"Hash Database Loaded Successfully. Number of Songs: {len(hash_database)}")

        # Inspect contents
        for song_id, hashes in hash_database.items():
            print(f"Song: {song_id}")
            print(f"Number of Hashes: {len(hashes)}")
            print(f"Sample Hashes: {hashes[:5]}")  # Print first 5 hashes for each song
            print("-" * 40)

    except FileNotFoundError:
        print(f"Database file '{database_file}' not found.")
    except Exception as e:
        print(f"Error loading database: {e}")





def record_audio(output_file="live_sample.wav", duration=5, sample_rate=44100):
    """
    Records audio from the microphone and saves it to a file.

    Args:
        output_file (str): Path to save the recorded audio file.
        duration (int): Duration of the recording in seconds.
        sample_rate (int): Sampling rate for the recording.
    """
    print("Recording started...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    write(output_file, sample_rate, audio_data)
    print(f"Recording finished. Saved to {output_file}")


def match_sample_to_database(live_sample="live_sample.wav", database_file="hash_database.pkl"):
    """
    Matches a live audio sample to the hash database.

    Args:
        live_sample (str): Path to the recorded sample.
        database_file (str): Path to the hash database.

    Returns:
        str: The name of the matching song or "No match found."
    """
    # Generate spectrogram and hashes for the live sample
    f, t, Zxx = generate_spectrogram(live_sample, nperseg=2048, noverlap=1024)
    key_points = extract_key_points_shazam(Zxx, threshold=0.3, neighborhood_size=(30, 30))
    anchor_points = choose_anchor_points(
        key_points, freq_bin_size=500, time_bin_size=10,
        max_freq_distance=3000, min_time_distance=1, max_time_distance=15
    )
    live_hashes = hash_anchor_points(anchor_points)

    # Debugging: Print live hashes
    print(f"Live hashes: {live_hashes[:10]}")  # Print first 10 live hashes for debugging

    # Load hash database
    with open(database_file, "rb") as db:
        hash_database = pickle.load(db)

    # Match hashes
    matches = {}
    for song_id, song_hashes in hash_database.items():
        common_hashes = set(live_hashes).intersection(song_hashes)
        matches[song_id] = len(common_hashes)
        # Debugging: Print common hashes for each song
        print(f"Song: {song_id}, Common Hashes: {len(common_hashes)}")

    # Find the best match
    if matches:
        best_match = max(matches, key=matches.get)
        print(f"Best match: {best_match} ({matches[best_match]} common hashes)")
        return best_match
    else:
        print("No match found.")
        return "No match found."


# Import functions from your project (e.g., generate_spectrogram, etc.)
# from your_project_file import generate_spectrogram, extract_key_points_shazam, choose_anchor_points, hash_anchor_points, match_sample_to_database


if __name__ == '__main__': 

    
    # Create a lock for synchronizing database updates
    # database_lock = threading.Lock()

    song_dir = 'labs/database'

    # Process each song without threading for simplicity
    song_files = [
        os.path.join(song_dir, file) for file in os.listdir(song_dir) if os.path.isfile(os.path.join(song_dir, file))
    ]



    # Process each song in its own thread
    # threads = []
    for song in song_files:
        thread = process_and_store_song_in_thread(song)

    # # Wait for all threads to finish
    # for thread in threads:
    #     thread.join()

    song_dir = 'labs/database'

    # Process each song without threading for simplicity
    song_files = [
        os.path.join(song_dir, file) for file in os.listdir(song_dir) if os.path.isfile(os.path.join(song_dir, file))
    ]



    # Process each song in its own thread
    # threads = []
    for song in song_files:
        thread = process_and_store_song_in_thread(song)

    # # Wait for all threads to finish
    # for thread in threads:
    #     thread.join()

    print("All songs have been processed and stored.")


    # Inspect the final hash database
    try:
        with open("hash_database.pkl", "rb") as db:
            hash_database = pickle.load(db)

        print(f"Final Hash Database: {len(hash_database)} songs")
        for song_id in hash_database:
            print(f"{song_id}: {len(hash_database[song_id])} hashes")
    except FileNotFoundError:
        print("Hash database not found. No songs were saved.")

    # Example usage
    inspect_hash_database("hash_database.pkl")

    # # Step 6: Match a live sample against the database
    # live_sample_path = 'labs/database/' + os.listdir(song_dir)[1]
    # print('live sample path: ', live_sample_path)


    # # Process the live sample
    # f, t, Zxx = generate_spectrogram(live_sample_path, nperseg=2048, noverlap=1024)
    # key_points = extract_key_points_shazam(Zxx, threshold=0.3, neighborhood_size=(30, 30))
    # anchor_points = choose_anchor_points(
    #     key_points,
    #     freq_bin_size=500,
    #     time_bin_size=10,
    #     max_freq_distance=5000,
    #     min_time_distance=1,
    #     max_time_distance=15
    # )
    # live_hashes = hash_anchor_points(anchor_points)

    # # Match against the database
    # best_match = match_hashes(live_hashes)
    # if best_match:
    #     print(f"The live recording matches: {best_match}")
    # else:
    #     print("No match found!")


    # Step 1: Record audio
    record_audio(output_file="test_sample.wav", duration=30)
    print("Recording completed. Saved as 'test_sample.wav'.")

    # Step 2: Match the recorded sample
    match_result = match_sample_to_database(live_sample='test_sample.wav', database_file="hash_database.pkl")
    print(f"Match result: {match_result}")

