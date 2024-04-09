import argparse
import os
import json


import sys
sys.path.append("./src")

import torchvision.models as models
from src.FeatureExtraction import get_feature_vector

def load_model(model_name:str):
    try:
        # Check if the model name is available in torchvision.models
        model_func = getattr(models, model_name)
        model = model_func(weights="DEFAULT")
        return model
    except AttributeError:
        print(f"Error: Model '{model_name}' not found.")
        return None

    

def main():

    parser = argparse.ArgumentParser(description='Load a PyTorch model.')
    parser.add_argument('model_name', type=str, help='Name of the model to load (e.g., resnet18, resnet50)')
    parser.add_argument('layer', type=str, help="Layer from which to extract features")
    parser.add_argument('image_dir', type=str, help="Directory of images to process")

    args = parser.parse_args()

    model = load_model(args.model_name)
    if model:
        print(f"Successfully loaded '{args.model_name}'.")

    if hasattr(model, args.layer):
        print(f"{args.layer} exists in model, proceeding...")
    else:
        raise AttributeError(f"{args.layer} does not exist in {args.model_name}")
    
    if os.path.isdir(args.image_dir):
        print(f"Proceeding to extract features from images in {args.image_dir}")
    else:
        raise NotADirectoryError(f"{args.image_dir} is not a valid directory")
    
    data = {}
    
    for item in os.listdir(args.image_dir):
        # Construct the full path
        item_path = os.path.join(args.image_dir, item)
        # Check if it's a file
        if os.path.isfile(item_path):
            # Do something with the file
            output = get_feature_vector(item_path, model, args.layer)
            data[item] = output.detach().numpy().tolist()
            
        else:
            print("Not a file:", item_path)
            continue

    fname = f"{args.model_name}.{args.layer}_features.json"
    with open(fname, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print(f"Feature Vectors written to file {fname}")
    
if __name__ == "__main__":
    main()
