import os
import random
import shutil

def extract_random_line(input_file, output_file):
    """
    Extracts a random line from input_file and saves it to output_file.
   
    Args:
        input_file (str): Path to the input text file
        output_file (str): Path to the output text file
    """
    try:
        # Read all lines from the input file
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
       
        # Check if the file has any lines
        if not lines:
            print(f"Warning: {input_file} is empty. No line extracted.")
            return
       
        # Select a random line
        random_line = random.choice(lines)
       
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Write the random line to the output file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(random_line)
       
        print(f"Successfully extracted a random line from {input_file} to {output_file}")
   
    except Exception as e:
        print(f"Error processing {input_file}: {e}")

def process_emotion_txt_files(input_dir='emotion', output_dir='extracted_emotions'):
    """
    Processes all .txt files in the given emotion directory and saves the random lines
    to a new directory with the same filenames.
   
    Args:
        input_dir (str): Directory to search for .txt files. Default is 'emotion'.
        output_dir (str): Directory to save the output files. Default is 'extracted_emotions'.
    """
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Input directory '{input_dir}' does not exist.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all .txt files in the input directory
    txt_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
   
    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return
   
    print(f"Found {len(txt_files)} .txt files to process")
   
    # Process each .txt file
    for txt_file in txt_files:
        input_path = os.path.join(input_dir, txt_file)
        output_path = os.path.join(output_dir, txt_file)
       
        extract_random_line(input_path, output_path)

if __name__ == "__main__":
    # Process all txt files from the emotion folder
    input_directory = "/home/fast/Documents/EmoCLIP-master/emotion"  # Change this to your emotion folder path
    output_directory = "/home/fast/Documents/EmoCLIP-master/extracted_emotions" 
    process_emotion_txt_files()
    print("Processing complete!")
