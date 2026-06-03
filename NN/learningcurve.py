import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# 1. Configurazione del dispositivo (usa la GPU se disponibile, es. su Colab)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo in uso: {device}")

# 2. Generazione di un dataset fittizio per classificazione binaria
np.random.seed(42)
torch.manual_seed(42)

# 1200 campioni, 10 feature ciascuno
X = np.random.randn(1200, 10).astype(np.float32)
# Generiamo i target (0 o 1) in base a una semplice regola lineare + rumore
y = (X[:, 0] + X[:, 1] > 0).astype(np.float32).reshape(-1, 1)

# Suddivisione in Train (80%) e Validation (20%)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Conversione in tensori PyTorch e spostamento sul dispositivo (CPU o GPU)
X_train_t = torch.tensor(X_train).to(device)
y_train_t = torch.tensor(y_train).to(device)
X_val_t = torch.tensor(X_val).to(device)
y_val_t = torch.tensor(y_val).to(device)

# 3. Definizione di una semplice rete neurale
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(10, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid() # Per classificazione binaria
        )
        
    def forward(self, x):
        return self.network(x)

model = SimpleNN().to(device)
criterion = nn.BCELoss() # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. Liste per salvare la "storia" dell'addestramento (i dati per la Learning Curve)
history = {
    'train_loss': [],
    'val_loss': [],
    'train_acc': [],
    'val_acc': []
}

epochs = 100

# 5. Loop di addestramento
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass (Train)
    train_outputs = model(X_train_t)
    train_loss = criterion(train_outputs, y_train_t)
    
    # Backward pass e ottimizzazione
    train_loss.backward()
    optimizer.step()
    
    # Calcolo accuratezza Train
    train_preds = (train_outputs > 0.5).float()
    train_acc = (train_preds == y_train_t).float().mean().item()
    
    # Validazione (senza calcolare i gradienti)
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_t)
        val_loss = criterion(val_outputs, y_val_t)
        val_preds = (val_outputs > 0.5).float()
        val_acc = (val_preds == y_val_t).float().mean().item()
        
    # Salvataggio delle metriche nella history
    history['train_loss'].append(train_loss.item())
    history['val_loss'].append(val_loss.item())
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    
    # Stampa l'avanzamento ogni 10 epoche
    if (epoch + 1) % 10 == 0:
        print(f"Epoca [{epoch+1}/{epochs}] -> Train Loss: {train_loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Val Acc: {val_acc:.4f}")

# 6. PLOTTING DELLA LEARNING CURVE
plt.figure(figsize=(14, 5))

# Grafico della Loss (Funzione di perdita)
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss', color='royalblue', linewidth=2)
plt.plot(history['val_loss'], label='Validation Loss', color='orange', linewidth=2, linestyle='--')
plt.title('Andamento della Loss (Learning Curve)', fontsize=14)
plt.xlabel('Epoche', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)

# Grafico dell'Accuracy (Accuratezza)
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Accuracy', color='mediumseagreen', linewidth=2)
plt.plot(history['val_acc'], label='Validation Accuracy', color='tomato', linewidth=2, linestyle='--')
plt.title('Andamento dell\'Accuracy', fontsize=14)
plt.xlabel('Epoche', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()


plt.show()