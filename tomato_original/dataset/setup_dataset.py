import os
import csv
import yaml

# Bird Classes for ScareX
BIRD_CLASSES = [
    "Crow", "Common_Myna", "Rose_Ringed_Parakeet", "Peacock", 
    "Pigeon", "Hen", "Sparrow", "Dove", "Koel", 
    "Duck", "Goose", "Turkey"
]

# Negative Classes for Audio
NEGATIVE_AUDIO_CLASSES = [
    "Wind", "Rain", "Tractor", "Motorcycle", "Dog", "Cow", 
    "Goat", "Human Voices", "Leaves", "Silence", "Water Flow", "Insects"
]

def create_directory_structure(base_path="ScareX_Dataset"):
    print(f"Creating directory structure in '{base_path}'...")
    
    # Subdirectories for splits
    splits = ["train", "valid", "test"]
    
    # Main categories
    categories = {
        "images": splits,
        "labels": splits,
        "audio": splits,
        "metadata": []
    }
    
    for category, subs in categories.items():
        if subs:
            for split in subs:
                os.makedirs(os.path.join(base_path, category, split), exist_ok=True)
        else:
            os.makedirs(os.path.join(base_path, category), exist_ok=True)
            
    print("Directory structure created successfully.")

def generate_yolo_yaml(base_path="ScareX_Dataset"):
    yaml_path = os.path.join(base_path, "data.yaml")
    
    data = {
        "train": "../train/images",
        "val": "../valid/images",
        "test": "../test/images",
        "nc": len(BIRD_CLASSES),
        "names": BIRD_CLASSES
    }
    
    with open(yaml_path, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False, sort_keys=False)
        
    print(f"YOLO data.yaml generated at {yaml_path}")

def create_metadata_templates(base_path="ScareX_Dataset"):
    metadata_dir = os.path.join(base_path, "metadata")
    
    # Image Metadata
    img_csv = os.path.join(metadata_dir, "image_metadata.csv")
    with open(img_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Species", "Time_of_Day", "Weather", "Environment", "Lighting", "is_synthetic"])
    
    # Audio Metadata
    aud_csv = os.path.join(metadata_dir, "audio_metadata.csv")
    with open(aud_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Species", "Duration_sec", "Sample_Rate", "Location_Type", "Noise_Level", "Weather", "Distance", "Time_of_Day", "is_synthetic"])
        
    print(f"Metadata CSV templates generated in {metadata_dir}")

def generate_synthetic_data_guidelines(base_path="ScareX_Dataset"):
    guideline_path = os.path.join(base_path, "SYNTHETIC_DATA_GUIDE.md")
    content = """# ScareX Synthetic Data Guidelines

To improve the robustness of models without extensive field data collection, follow these synthetic generation strategies:

## 1. Image Compositing
- **Foreground:** Segment birds from clear images (using SAM or similar tools).
- **Background:** Collect images of agricultural fields (Paddy, Sugarcane) under various lighting conditions.
- **Blending:** Paste the foreground onto the background. Use Poisson blending or simple alpha compositing with random scaling.
- **Labeling:** Automatically generate bounding boxes based on the pasted foreground mask.

## 2. Audio Mixing
- **Clean Signal:** Use high-quality bird calls.
- **Noise:** Use negative class recordings (Wind, Tractor, Rain).
- **Mixing:** Overlay the clean signal with noise at various SNR (Signal-to-Noise Ratio) levels (e.g., -5dB, 0dB, +5dB).
- **Export:** Export as 16kHz mono `.wav`.

*Note: Always flag synthetic data in the metadata CSVs with `is_synthetic=True` and append `_synth` to the filename.*
"""
    with open(guideline_path, 'w') as file:
        file.write(content)
    print(f"Synthetic data guidelines created at {guideline_path}")

if __name__ == "__main__":
    # Create dataset in the parent directory of this script or specified location
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "ScareX_Dataset")
    
    create_directory_structure(base_dir)
    generate_yolo_yaml(base_dir)
    create_metadata_templates(base_dir)
    generate_synthetic_data_guidelines(base_dir)
    
    print("\nDataset generation structure setup complete!")
