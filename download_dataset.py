import gdown
import os

def load_data():
    url = 'https://drive.google.com/drive/folders/1tGh4B1oM3dRjTiFc12BPY3GMi3shu3Tf'
    output_dir = 'd:/Projects/DSAI_490/Assignments/A1/data/raw'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Downloading dataset from Google Drive...")
    try:
        # remaining_ok=True allows gdown to proceed even if it hits the 50 file limit per folder
        gdown.download_folder(url, output=output_dir, quiet=False, use_cookies=False, remaining_ok=True)
        print("Download finished.")
    except Exception as e:
        print(f"Error during download: {e}")

if __name__ == '__main__':
    load_data()
