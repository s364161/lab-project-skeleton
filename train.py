import time
import torch

from torch import nn
from network.py import CustomNet
from data.tiny_imagenet import get_loaders

### RIPRENDI IL TUO LAVORO DA QUI
# guarda la lezione minuto 11:42 per capire se mettere train e val def dentro utils oppure come funzioni del train.py
# scopri inoltre eval.py a che cosa serve
# prova a runnare il codice su colab
# inserisci una visualizzazione con wandb e poi hai finito con il lab3
#  wandb devi inserirlo nei requirements.txt e poi sul colab chiami !pip install requirements.txt

if __name__ == "__main__": # RICORDA DI FARE PUSH ALLA FINE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Lavoro su: {device}")

    train_loader, val_loader = get_loaders(batch_size=32)

    model = CustomNet().cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    best_acc = 0

    # Run the training process for {num_epochs} epochs
    num_epochs = 5
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        train(epoch, model, train_loader, criterion, optimizer)

        # At the end of each training iteration, perform a validation step
        val_accuracy = validate(model, val_loader, criterion)

        end_time = time.time()
        epoch_time = end_time - start_time
        print(f"Epoch duration: {epoch_time:2f} seconds")
        print()

        # Best validation accuracy
        best_acc = max(best_acc, val_accuracy)


    print(f'Best validation accuracy: {best_acc:.2f}%')
