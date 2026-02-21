import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score, f1_score, recall_score
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class BinaryMLP(nn.Module):
    def __init__(self, in_dim, num_classes=2, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024), 
            nn.ReLU(), 
            nn.Dropout(drop),
            nn.Linear(1024, 512), 
            nn.ReLU(), 
            nn.Dropout(drop),
            nn.Linear(512, 256), 
            nn.ReLU(), 
            nn.Dropout(drop),
            nn.Linear(256, 128), 
            nn.ReLU(), 
            nn.Dropout(drop),
            nn.Linear(128, 64), 
            nn.ReLU(), 
            nn.Dropout(drop),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

def to_tensors(X_df, y_arr=None):
    X_t = torch.from_numpy(np.asarray(X_df, dtype=np.float32))
    if y_arr is not None:
        y_t = torch.from_numpy(np.asarray(y_arr, dtype=np.int64))
        return X_t, y_t
    return X_t
def eval_loop(model, loader, le):

    model.eval()
    all_logits = []
    all_y = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            all_logits.append(logits.cpu())
            all_y.append(yb)
    logits_all = torch.cat(all_logits)
    y_all = torch.cat(all_y)
    probs = torch.softmax(logits_all, dim=1).numpy()
    preds = probs.argmax(axis=1)
    acc = accuracy_score(y_all, preds)
    f1  = f1_score(y_all, preds, average='binary' if probs.shape[1]==2 else 'weighted')
    recall = recall_score(y_all, preds, average='binary' if probs.shape[1]==2 else 'weighted')
    if probs.shape[1] == 2:
        auc = roc_auc_score(y_all, probs[:,1])
    else:
        auc = roc_auc_score(y_all, probs, multi_class='ovr', average='weighted')

    conf = confusion_matrix(y_all, preds, labels=range(len(le.classes_)))
    report = classification_report(y_all, preds, target_names=le.classes_, digits=3)

    return {'acc': acc, 'f1': f1, 'recall': recall, 'auc': auc, 'conf': conf, 'report': report, 'pred_y': preds}


def train_neural_network(X_train, y_train, le, input_dim, epochs=20, batch_size=256, lr=1e-3, patience=5):
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    X_train_t, y_train_t = to_tensors(X_train_sub, y_train_sub)
    X_val_t, y_val_t = to_tensors(X_val, y_val)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=512, shuffle=False) # Batch size for eval can be larger

    model = BinaryMLP(in_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_val_auc = -np.inf
    wait = 0
    best_model_state = None

    print(f"Start NN Training on {device}...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        
        train_loss = total_loss / len(train_loader.dataset)
        val_metrics = eval_loop(model, val_loader, le)
        print(f'Epoch {epoch:02d}/{epochs} | loss {train_loss:.4f} | '
              f'val_acc {val_metrics["acc"]:.3f} | val_f1 {val_metrics["f1"]:.3f} | val_auc {val_metrics["auc"]:.3f}')

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            wait = 0
            best_model_state = model.state_dict()
        else:
            wait += 1
            if wait >= patience:
                print('Early stopping triggered.')
                break
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    return model

def predict_nn(model, X_test):
    model.eval()
    X_test_t = to_tensors(X_test)
    X_test_t = X_test_t.to(device)
    with torch.no_grad():
        logits = model(X_test_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs