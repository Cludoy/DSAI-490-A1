import os
import sys

def main():
    # print("Step 1: Downloading data...")
    # # Import and run the download logic
    # try:
    #     import download_dataset
    #     download_dataset.load_data()
    # except Exception as e:
    #     print(f"Warning during download: {e}")
    #     print("We will proceed with whatever data was downloaded.")
        
    print("\nStep 2: Training models...")
    # Add src to path so we can import train
    sys.path.append(os.path.abspath('src'))
    try:
        from train import train_models
        train_models()
        print("\nPipeline completed successfully!")
    except Exception as e:
        print(f"Error during training pipeline: {e}")

if __name__ == '__main__':
    main()
