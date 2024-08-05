import os
from PIL import Image
from pathlib import Path

def convert_rgba_to_rgb(input_path, output_path):
    with Image.open(input_path) as img:
        # Convert to RGB
        rgb_img = img.convert('RGB')
        # Save the image
        rgb_img.save(output_path)

def process_directory(input_directory, output_directory):
    # Expand the ~ to the full home directory path
    input_directory = os.path.expanduser(input_directory)
    output_directory = os.path.expanduser(output_directory)
    
    # Create Path objects
    input_dir_path = Path(input_directory)
    output_dir_path = Path(output_directory)
    
    # Create the output directory if it doesn't exist
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Iterate over all files in the input directory
    for file_path in input_dir_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            print(f"Processing {file_path}")
            
            # Create output path
            output_path = output_dir_path / file_path.name
            
            try:
                # Convert the image
                convert_rgba_to_rgb(file_path, output_path)
                print(f"Converted {file_path.name} to RGB and saved in {output_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")

if __name__ == "__main__":
    input_directory = "/vol/fob-vol3/mi20/deghaisa/code/shot_0003/1_source_sequence"
    output_directory = "/vol/fob-vol3/mi20/deghaisa/code/shot_0003/1_source_sequence_rgb"
    process_directory(input_directory, output_directory)
    print("All images processed.")