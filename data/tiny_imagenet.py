import os
import shutil
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

def get_loaders(batch_size=32):
    # --- Il tuo codice di riorganizzazione (Modifica minima: controllo if) ---
    data_root = 'dataset/tiny-imagenet/tiny-imagenet-200'
    val_path = os.path.join(data_root, 'val')
    
    # Eseguiamo la riorganizzazione solo se la cartella 'images' esiste ancora
    if os.path.exists(os.path.join(val_path, 'images')):
        with open(os.path.join(val_path, 'val_annotations.txt')) as f:
            for line in f:
                fn, cls, *_ = line.split('\t')
                os.makedirs(os.path.join(val_path, cls), exist_ok=True)
                shutil.copyfile(os.path.join(val_path, 'images', fn), os.path.join(val_path, cls, fn))
        shutil.rmtree(os.path.join(val_path, 'images'))

    # --- Il tuo codice per trasformazioni e dataset ---
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_set = ImageFolder(root=os.path.join(data_root, 'train'), transform=transform)
    val_set = ImageFolder(root=val_path, transform=transform)

    # --- I tuoi DataLoader ---
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader