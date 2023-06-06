import torch
import glob
import os
from PIL import Image
import random
from torchvision import transforms

def file_list(data_dir):
    list_label = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    file_list = []
    for i in list_label:
        file_list += glob.glob(os.path.join(data_dir + '/' + i,'*.jpg'))
    return file_list

def save_model(args, model):
    folder_path = args.model_folder
    model_name = args.model
    i = 1
    new_model_name = model_name
    while os.path.exists(os.path.join(folder_path, new_model_name + ".pt")):
        new_model_name = model_name + "_" + str(i)
        i += 1

    path = os.path.join(folder_path, new_model_name + ".pt")

    torch.save(model, path)

    print(f"Model saved to {new_model_name}.pt")

def save_checkpoint(model, optimizer, epoch, loss, args):
    checkpoint_path = os.path.join(args.model_folder, args.model + '_' + str(epoch) + ".pt")
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

def load_model(model, optimizer, checkpoint_path=None):
    if checkpoint_path is None:
        # Load from scratch
        return model, optimizer, 0

    # Load from checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    print(f"Loaded checkpoint from epoch {epoch}")

    return model, optimizer, epoch

def predict_image(image_path, model, args):
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                            std = [0.229, 0.224, 0.225]),
    ])

    class_map = {"daisy" : 0, "dandelion": 1,
        "roses" : 2, "sunflowers": 3, "tulips" : 4}
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0).to(args.device)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_idx = torch.max(outputs, 1)
    predicted_label = predicted_idx.item()
    predicted_class = [key for key, value in class_map.items() if value == predicted_label][0]

    return predicted_class


def display_image(image_path, predicted_label):
    image = Image.open(image_path)
    image.show()
    print(f"Predicted label: {predicted_label}")

def predict_folder(folder_path):
    image_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(image_paths)
    selected_images = image_paths[:15]
    rows, cols = 3, 5
    for i in range(rows):
        for j in range(cols):
            index = i * cols + j
            if index < len(selected_images):
                image_path = selected_images[index]
                predicted_label = predict_image(image_path)
                display_image(image_path, predicted_label)
                print()
            else:
                break