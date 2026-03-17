import argparse
import torch
import random
import numpy as np
from torchvision.transforms import v2
from hyperparameters import HyperParameters
from train import DenseNetworkModel

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", action="store_true", default=False, help="Enable logging")
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        print("mps")
        device = torch.device("mps")
    elif torch.cuda.is_available():
        print("cuda")
        device = torch.device("cuda")
    else:
        print("cpu")
        device = torch.device("cpu")

    root_directory = ""

    # Hyperparams
    hyperparams = HyperParameters(num_epochs=100, batch_size=128, learning_rate=0.1, momentum=0.9, weight_decay=1e-4)

    train_transforms = v2.Compose([  
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float, scale=True),
        v2.ColorJitter(brightness=0.1, contrast=0.1),
        v2.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
    ])

    validation_transforms = v2.Compose([
        v2.ToDtype(torch.float, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], 
                     std=[0.229, 0.224, 0.225])
    ])

    densenet = DenseNetworkModel(device, log=args.log)
    densenet.train(root_directory, hyperparams, train_transforms, validation_transforms)

# logging off by default
# for logging run "uv run main.py --log"  
if __name__ == "__main__":
    main()
