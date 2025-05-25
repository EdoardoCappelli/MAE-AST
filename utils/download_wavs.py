# import os
# import time
# import glob
# import datetime 
# import argparse
# import subprocess

# def download_wavs(args):
#     """Download videos and extract audio in wav format.
#     """

#     # Paths
#     csv_path = args.csv_path
#     audios_dir = args.audios_dir
#     mini_data = args.mini_data
    
#     if not os.path.exists(audios_dir):
#         os.makedirs(audios_dir)
    
#     # Read csv
#     with open(csv_path, 'r') as f:
#         lines = f.readlines()
    
#     lines = lines[3:]   # Remove csv head info

#     if mini_data:
#         lines = lines[0 : 10]   # Download partial data for debug
    
#     # Download
#     for (n, line) in enumerate(lines):
        
#         items = line.split(', ')
#         audio_id = items[0]
#         start_time = float(items[1])
#         end_time = float(items[2])
#         duration = end_time - start_time
        
#         # Download full video of whatever format
#         video_name = audios_dir + "/" + f"_Y{audio_id}"

#         print(f"Downloading {n + 1}/{len(lines)}: {audio_id} ...")
#         # subprocess.run([
#         #     "yt-dlp",
#         #     "--quiet",
#         #     "-o", video_name,
#         #     "-x",              # estrai l’audio in formato originale
#         #     f"https://www.youtube.com/watch?v={audio_id}"
#         # ], check=False)
#         subprocess.run([
#             "yt-dlp",
#             "-f", "bestaudio",
#             "--quiet",
#             "--extract-audio",
#             "--audio-format", "wav",
#             "-o", video_name,
#             f"https://www.youtube.com/watch?v={audio_id}"
#         ], check=False)
                
#         audio_path = audios_dir + "/" + '_Y' + audio_id + '.wav'

#         os.system("ffmpeg -y -loglevel panic -i {} -ac 1 -ar 32000 -ss {} -t 00:00:{} {} "\
#                 .format(video_name, 
#                 str(datetime.timedelta(seconds=start_time)), duration, 
#                 audio_path))
    
#         # Remove downloaded video
#         if os.path.exists(video_name):
#             os.remove(video_name)

  
# parser = argparse.ArgumentParser(description='Download wavs from csv file')
# parser.add_argument('--csv_path', type=str, default='VisualTransformer/datasets/csv/balanced_train_segments.csv', help='Path to the csv file')
# parser.add_argument('--audios_dir', type=str, default='VisualTransformer/datasets/wavs/balanced_train_segments', help='Path to the audio directory')
# parser.add_argument('--mini_data', action='store_true', help='Use mini data for debug')
# args = parser.parse_args()

# download_wavs(args)
import os
import time
import glob
import datetime 
import argparse
import subprocess
import multiprocessing
from functools import partial
from tqdm import tqdm

def process_audio(line, audios_dir):
    """Download and process a single audio file.
    """
    items = line.split(', ')
    audio_id = items[0]
    start_time = float(items[1])
    end_time = float(items[2])
    duration = end_time - start_time
    
    # Download full video of whatever format
    video_name = os.path.join(audios_dir, f"FULL_{audio_id}")
    audio_path = os.path.join(audios_dir, f"FULL_{audio_id}.wav")
    
    # Skip if the final audio file already exists
    if os.path.exists(audio_path):
        return audio_id, "Already exists"
    
    try:
        # Download audio in wav format
        result = subprocess.run([
            "yt-dlp",
            "-f", "bestaudio",
            "--quiet",
            "--extract-audio",
            "--audio-format", "wav",
            "-o", video_name,
            f"https://www.youtube.com/watch?v={audio_id}"
        ], check=False)
        
        if result.returncode != 0:
            return audio_id, f"Download failed with code {result.returncode}"
        
        # Extract segment with ffmpeg
        final_audio_path = os.path.join(audios_dir, f"EX_{audio_id}.wav")
        ffmpeg_cmd = f"ffmpeg -y -loglevel panic -i {audio_path} -ac 1 -ar 32000 -ss {str(datetime.timedelta(seconds=start_time))} -t 00:00:{duration} {final_audio_path}"
        os.system(ffmpeg_cmd)
        
        if os.path.exists(audio_path):
            os.remove(audio_path)

        return audio_id, "Success"
    except Exception as e:
        return audio_id, f"Error: {str(e)}"

def download_wavs(args):
    """Download videos and extract audio in wav format using parallel processing.
    """
    # Paths
    csv_path = args.csv_path
    audios_dir = args.audios_dir
    mini_data = args.mini_data
    num_workers = args.num_workers
    
    if not os.path.exists(audios_dir):
        os.makedirs(audios_dir)
    
    # Read csv
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    
    lines = lines[3:]  # Remove csv head info

    if mini_data:
        lines = lines[0:10]  # Download partial data for debug
    
    print(f"Processing {len(lines)} audio files with {num_workers} workers")
    
    # Create a partial function with fixed audios_dir
    process_func = partial(process_audio, audios_dir=audios_dir)
    
    # Use multiprocessing pool to process files in parallel
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(process_func, lines), total=len(lines), desc="Processing"))
    
    # Print summary
    successes = sum(1 for _, status in results if status == "Success")
    already_exists = sum(1 for _, status in results if status == "Already exists")
    failures = len(results) - successes - already_exists
    
    print(f"\nSummary:")
    print(f"  Successful downloads: {successes}")
    print(f"  Already existed: {already_exists}")
    print(f"  Failed downloads: {failures}")
    
    # Print failed downloads if any
    if failures > 0:
        print("\nFailed downloads:")
        for audio_id, status in results:
            if status != "Success" and status != "Already exists":
                print(f"  {audio_id}: {status}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download wavs from csv file in parallel')
    
    parser.add_argument('--csv_path', type=str, default='VisualTransformer/datasets/csv/balanced_train_segments.csv', help='Path to the csv file')
    
    parser.add_argument('--audios_dir', type=str, default='VisualTransformer/datasets/wavs/balanced_train_segments', help='Path to the audio directory')

    parser.add_argument('--mini_data', action='store_true', 
                        help='Use mini data for debug')
    parser.add_argument('--num_workers', type=int, default=os.cpu_count(), 
                        help='Number of parallel workers (default: number of CPU cores)')
    args = parser.parse_args()

    start_time = time.time()
    download_wavs(args)
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")