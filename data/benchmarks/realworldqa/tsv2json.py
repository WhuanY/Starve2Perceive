import os
import json
import base64

# Define the file paths
tsv_file = "RealWorldQA.tsv"
jsonl_file = "RealWorldQA.jsonl"
images_dir = "images"

# Ensure the images/ directory exists
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

# Open the TSV and process each line
with open(tsv_file, "r", encoding="utf-8") as tsv, open(jsonl_file, "w", encoding="utf-8") as jsonl:
    # Read the header and ignore it
    header = tsv.readline().strip().split("\t")
    
    # Process each line of the TSV file
    for line in tsv:
        fields = line.strip().split("\t")
        
        # Extract relevant fields
        image_base64 = fields[0].strip()  # Base64-encoded image data
        answer = fields[1].strip().strip("\"")  # A, B, C, or D
        index = fields[2].strip("\"")  # index field
        options = [fields[3].strip("\""), fields[4].strip("\""), fields[5].strip("\""), fields[6].strip("\"")]  # A, B, C, D options
        question = fields[7].strip("\"")  # question text
        
        # Remove empty options
        options = [opt for opt in options if opt]
        
        # Save the image to the images/ directory
        image_path = os.path.join(images_dir, f"{index}.jpg")
        with open(image_path, "wb") as image_file:
            image_file.write(base64.b64decode(image_base64))
        
        # Build the JSON object
        json_obj = {
            "index": index,
            "question": question,
            "options": options if len(options) > 1 else [],  # If not multiple choice, set options to an empty list
            "answer": answer,
            "image_path": image_path
        }
        
        # Write the JSON object to the output file
        jsonl.write(json.dumps(json_obj, ensure_ascii=False) + "\n")