import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from typing import Dict, List

def plot_regression_scatter(y_true: np.ndarray, preds_dict: Dict[str, np.ndarray], title: str = "Regresja: Wartości Rzeczywiste vs Przewidywane"):
    fig, axes = plt.subplots(1, len(preds_dict), figsize=(6 * len(preds_dict), 5))
    if len(preds_dict) == 1:
        axes = [axes]
        
    for ax, (name, y_pred) in zip(axes, preds_dict.items()):
        # Rysowanie punktów z przezroczystością, by lepiej widzieć zagęszczenie
        ax.scatter(y_true, y_pred, alpha=0.2, color='royalblue', edgecolors='none')
        
        # Rysowanie idealnej linii (y_true == y_pred)
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Idealna predykcja')
        
        ax.set_title(name, fontsize=14)
        ax.set_xlabel("Wartości Rzeczywiste", fontsize=12)
        ax.set_ylabel("Przewidywania", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        
    plt.suptitle(title, fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig("regression_scatter.png", bbox_inches='tight')
    plt.show()

def plot_confusion_matrices(y_true: np.ndarray, preds_dict: Dict[str, np.ndarray], classes: List[str], title: str = "Klasyfikacja: Macierze Pomyłek"):
    fig, axes = plt.subplots(1, len(preds_dict), figsize=(6 * len(preds_dict), 5))
    if len(preds_dict) == 1:
        axes = [axes]
        
    for ax, (name, y_pred) in zip(axes, preds_dict.items()):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        # Tworzenie wykresu na konkretnej osi (ax)
        disp.plot(ax=ax, cmap='Blues', colorbar=False)
        ax.set_title(name, fontsize=14)
        ax.grid(False) # Wyłączamy siatkę, by macierz była czytelna
        
    plt.suptitle(title, fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig("classification_cm.png", bbox_inches='tight')
    plt.show()

def plot_bar_comparison(metrics_dict: Dict[str, float], title: str, ylabel: str, filename: str):
    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    plt.figure(figsize=(8, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    bars = plt.bar(names, values, color=colors[:len(names)], edgecolor='black')
    
    plt.title(title, fontsize=14)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Dodanie wartości liczbowych nad słupkami
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(values)*0.02), 
                 f'{yval:.4f}', ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.show()