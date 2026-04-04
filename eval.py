import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from models.network import CustomNet
from data.tiny_imagenet import get_loaders

def evaluate_diagnostics(model, val_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    top1_correct = 0
    top5_correct = 0
    total = 0

    print("Inizio analisi dettagliata...")
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # --- Calcolo Top-1 e Top-5 Accuracy ---
            # sorted_indices ci dà le classi ordinate dalla più probabile alla meno
            _, max_k_preds = outputs.topk(5, dim=1, largest=True, sorted=True)
            
            total += targets.size(0)
            # Top-1: la prima colonna deve corrispondere al target
            top1_correct += (max_k_preds[:, 0] == targets).sum().item()
            # Top-5: il target deve essere presente in una delle 5 colonne
            top5_correct += (max_k_preds == targets.view(-1, 1)).sum().item()

            # --- Raccolta dati per Confusion Matrix ---
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Calcolo percentuali finali
    top1_acc = 100. * top1_correct / total
    top5_acc = 100. * top5_correct / total

    print(f"\n--- RISULTATI FINALI ---")
    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")

    # --- Plot Confusion Matrix (Semplificata) ---
    # Nota: TinyImageNet ha 200 classi, una matrice 200x200 è illeggibile.
    # Spesso si visualizza solo una porzione o si analizzano i dati numericamente.
    cm = confusion_matrix(all_targets, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm[:20, :20], annot=True, fmt='d', cmap='Blues') # Mostriamo solo le prime 20 classi
    plt.title("Confusion Matrix (Prime 20 classi)")
    plt.xlabel("Predetti")
    plt.ylabel("Reali")
    plt.savefig("confusion_matrix_subset.png")
    print("\nGrafico della Confusion Matrix (prime 20 classi) salvato come 'confusion_matrix_subset.png'")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Per farlo girare subito dopo il training in Colab, 
    # passeremo il modello già in memoria o lo ricaricheremo.
    # Qui carichiamo loaders e modello per test
    _, val_loader = get_loaders(batch_size=32)
    model = CustomNet().to(device) 
    
    # Se hai i pesi salvati: 
    # model.load_state_dict(torch.load('checkpoints/best_model.pth'))
    
    evaluate_diagnostics(model, val_loader, device)