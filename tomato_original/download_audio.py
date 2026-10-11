import urllib.request
import json
import os
import glob

def download_crow_audio():
    base_dir = os.path.dirname(__file__)
    crow_dir = os.path.join(base_dir, 'dataset', 'audio', 'birds', 'crow')
    os.makedirs(crow_dir, exist_ok=True)
    
    # Clean up old mock files
    for f in glob.glob(os.path.join(crow_dir, 'mock_*.wav')):
        os.remove(f)

    # Direct download from Google's free sound library
    url = "https://actions.google.com/sounds/v1/animals/crow_caw.ogg"
    filename = os.path.join(crow_dir, 'real_crow_0.ogg')
    print(f"Downloading {url} to {filename}...")
    
    req_dl = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req_dl) as response_dl, open(filename, 'wb') as out_file:
        out_file.write(response_dl.read())
        
    print("Successfully downloaded real crow audio file.")

if __name__ == "__main__":
    download_crow_audio()
