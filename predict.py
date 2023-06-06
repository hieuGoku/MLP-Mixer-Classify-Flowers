import torch
import os
from argparse import ArgumentParser
from utils.utils import *

if __name__ == "__main__":
    parser = ArgumentParser()
    home_dir = '/content'
    parser.add_argument("--source", default='{}/data/test'.format(home_dir), type=str)
    parser.add_argument("--image-size", default=256, type=int)
    parser.add_argument("--model", default='mixer_s32.pt', type=str, help='path/to/model')
    parser.add_argument("--device", default='cuda', type=str, help='cuda, cpu')

    args = parser.parse_args()

    # Load model
    model = torch.load(args.model).to(args.device)
    model.eval()

    input_path = args.source
    if os.path.isfile(input_path):
        predicted_label = predict_image(input_path, model, args)
        print(f"Predicted label: {predicted_label}")
    else:
        print("Invalid input path!")
