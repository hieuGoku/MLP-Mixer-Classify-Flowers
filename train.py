from argparse import ArgumentParser
import torch
import torch.nn as nn
from torchvision import transforms
import wandb
from models.mlp_mixer import *
from utils.data import *
from utils.utils import *

parser = ArgumentParser()
home_dir = '/content'
parser.add_argument("--logger", default=None)
parser.add_argument("--id-name", default=None, type=str)
parser.add_argument("--train-folder", default='{}/data/train'.format(home_dir), type=str)
parser.add_argument("--valid-folder", default='{}/data/valid'.format(home_dir), type=str)
parser.add_argument("--model-folder", default='{}/model'.format(home_dir), type=str)
parser.add_argument("--resume", default=None, type=str, help='weight to resume training')
parser.add_argument("--save-interval", default=None, type=int)
parser.add_argument("--model", default=None, type=str, help='mixer_s32, mixer_s16, mixer_b32, mixer_b16, mixer_l32, mixer_l16, mixer_h14')
parser.add_argument("--num-classes", default=5, type=int)
parser.add_argument("--num-mlp-blocks", default=8, type=int)
parser.add_argument("--patch-size", default=16, type=int)
parser.add_argument("--hidden-dim", default=512, type=int, help='Projection units')
parser.add_argument("--tokens-mlp-dim", default=256, type=int, help='Token-mixing units')
parser.add_argument("--channels-mlp-dim", default=2048, type=int, help='Channel-mixing units')
parser.add_argument("--image-size", default=256, type=int)
parser.add_argument("--batch-size", default=4, type=int)
parser.add_argument("--epochs", default=1, type=int)
parser.add_argument("--lr", default=0.0001, type=float)
parser.add_argument("--image-channels", default=3, type=int)
parser.add_argument("--device", default='cuda', type=str, help='cuda, cpu')

args = parser.parse_args()

def train_model(args, model):
    print('Training ...')

    if args.logger is not None:
        if args.id_name is not None:
            wandb.init(id=args.id_name, project='MLP-Mixer', resume="must")
        else:
            wandb.init(project='MLP-Mixer', name=args.model)

        wandb.watch(model)

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0
        epoch_accuracy = 0

        for data, label in train_loader:
            data = data.to(args.device)
            label = label.squeeze().type(torch.LongTensor).to(args.device)
            output = model(data)
            
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()

            epoch_accuracy += acc / len(train_loader)

            epoch_loss += loss / len(train_loader)

            torch.cuda.empty_cache()
        
        train_accuracy.append(epoch_accuracy.item())
        train_losses.append(epoch_loss.item())

        print(
            "Epoch : {}, train accuracy : {}, train loss : {}".format(
                epoch + 1, epoch_accuracy, epoch_loss
            )
        )

        if args.logger is not None:
            wandb.log({'Train Accuracy': epoch_accuracy.item(), 'Train Loss': epoch_loss.item(), 'Epoch': epoch + 1})

        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in valid_loader:

                data = data.to(args.device)
                label = label.squeeze().type(torch.LongTensor).to(args.device)

                val_output = model(data)
                val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().mean()

                epoch_val_accuracy += acc / len(valid_loader)

                epoch_val_loss += val_loss / len(valid_loader)

            val_accuracy.append(epoch_val_accuracy.item())
            val_losses.append(epoch_val_loss.item())
            print(
                "Epoch : {}, val_accuracy : {}, val_loss : {}".format(
                    epoch + 1, epoch_val_accuracy, epoch_val_loss
                )
            )
            if args.logger is not None:
                wandb.log({'Val Accuracy': epoch_val_accuracy.item(), 'Val Loss': epoch_val_loss.item(), 'Epoch': epoch + 1})
        if save_interval is not None:
            if (epoch+1) % save_interval == 0:
                save_checkpoint(model, optimizer, epoch+1, loss, args)

    if args.logger is not None:
        wandb.finish()
        
    save_model(args, model)

# data Augumentation
train_transforms =  transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomResizedCrop(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=30),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                            std = [0.229, 0.224, 0.225]),
    ])

val_transforms = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                            std = [0.229, 0.224, 0.225]),
    ])

train_data = dataset(file_list(args.train_folder), transforms=train_transforms)
valid_data = dataset(file_list(args.valid_folder), transforms=val_transforms)

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=valid_data, batch_size=args.batch_size, shuffle=True)

models_mapping = {
    'mixer_s32': mixer_s32,
    'mixer_s16': mixer_s16,
    'mixer_b32': mixer_b32,
    'mixer_b16': mixer_b16,
    'mixer_l32': mixer_l32,
    'mixer_l16': mixer_l16,
    'mixer_h14': mixer_h14
}

if args.model is not None:
    model = models_mapping[args.model](num_classes=args.num_classes, image_size=args.image_size).to(args.device)
elif args.resume is not None:
    args.model = args.resume.split('/')[-1].split('.')[0]
else:
    args.model = 'custom'
    model = MlpMixer(args.num_classes, args.num_blocks, args.patch_size, 
            args.hidden_dim, args.tokens_mlp_dim, args.channels_mlp_dim, args.image_size).to(args.device)

optimizer = torch.optim.Adam(params = model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()

start_epoch = 0
epochs = args.epochs
save_interval = args.save_interval

running_loss_train = 0
running_accuracy_train = 0
running_loss_val = 0
running_accuracy_val = 0

train_losses = []
val_losses = []
train_accuracy = []
val_accuracy = []

model, optimizer, start_epoch = load_model(model, optimizer, checkpoint_path=args.resume)

if __name__ == "__main__":
    train_model(args, model)

