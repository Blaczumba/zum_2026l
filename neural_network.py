import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler

class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.net(x)

class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)

def train_nn_regressor(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, epochs=100):
    """
    Train a neural network regressor with proper normalization and regularization.
    
    Args:
        X_train: Training features (already scaled)
        y_train: Training targets
        X_test: Test features
        epochs: Number of training epochs (default 100)
    
    Returns:
        model: Trained PyTorch model
        preds: Predictions on test set
    """
    # Normalize target values for better neural network training
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).squeeze()
    
    model = MLPRegressor(X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)

    X_tr_t = torch.FloatTensor(X_train)
    y_tr_t = torch.FloatTensor(y_train_scaled).unsqueeze(1)

    dataset = TensorDataset(X_tr_t, y_tr_t)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            out = model(batch_X)
            loss = criterion(out, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        epoch_loss /= len(loader)
        scheduler.step(epoch_loss)
        
        # Early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.eval()
    with torch.no_grad():
        preds_scaled = model(torch.FloatTensor(X_test)).numpy().squeeze()
    
    # Inverse transform predictions back to original scale
    preds = y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).squeeze()
    
    return model, preds

def train_nn_classifier(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, epochs=100):
    """
    Train a neural network classifier with proper regularization.
    
    Args:
        X_train: Training features (already scaled)
        y_train: Training targets
        X_test: Test features
        epochs: Number of training epochs (default 100)
    
    Returns:
        model: Trained PyTorch model
        preds: Predictions on test set
    """
    num_classes = len(np.unique(y_train))
    model = MLPClassifier(X_train.shape[1], num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)

    X_tr_t = torch.FloatTensor(X_train)
    y_tr_t = torch.LongTensor(y_train)

    dataset = TensorDataset(X_tr_t, y_tr_t)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            out = model(batch_X)
            loss = criterion(out, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        epoch_loss /= len(loader)
        scheduler.step(epoch_loss)
        
        # Early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(X_test))
        preds = torch.argmax(logits, dim=1).numpy()
    
    return model, preds
