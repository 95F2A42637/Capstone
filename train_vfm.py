"""
Phase 2 Training Script: Lightweight DINOv2 Classification Head on Cached Features.

Features:
- Loads cached representations from outputs/features/
- Trains DINOv2ClassificationHead (384 -> LayerNorm -> Linear(384->256) -> GELU -> Dropout -> Linear(256->2))
- AdamW optimizer (lr=1e-4, weight_decay=1e-4)
- Evaluates on Validation split every epoch, saves best checkpoint based on Val Accuracy
- Evaluates final model on held-out Test split
- Generates:
  - models/dinov2_authenticity_head.pth
  - outputs/training_config.json
  - outputs/dinov2_test_report.txt
  - outputs/dinov2_confusion_matrix.csv
  - outputs/dinov2_confusion_matrix.png
  - outputs/dinov2_loss_curve.png
  - outputs/dinov2_accuracy_curve.png
"""

import os
import time
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from app.models_vfm import DINOv2ClassificationHead, EMBEDDING_DIM


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cached_data(cache_dir: str, split: str):
    file_path = os.path.join(cache_dir, split, "features.pt")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cached features file not found: {file_path}. Run cache_features.py first.")
    payload = torch.load(file_path, weights_only=False)
    return payload["features"], payload["labels"], payload["classes"]


def train_classifier(args):
    set_seed(args.seed)
    device = torch.device("cpu")
    print("=" * 70)
    print("PHASE 2: TRAINING DINOv2 LIGHTWEIGHT CLASSIFICATION HEAD")
    print(f"Device: {device} | Seed: {args.seed} | Epochs: {args.epochs} | LR: {args.lr}")
    print("=" * 70)

    # 1. Load cached features
    print("\n[Step 1] Loading cached features...")
    train_x, train_y, classes = load_cached_data(args.cache_dir, "train")
    val_x, val_y, _ = load_cached_data(args.cache_dir, "val")
    test_x, test_y, _ = load_cached_data(args.cache_dir, "test")

    print(f"    Train Samples: {len(train_y):,} (Authentic: {(train_y==0).sum().item()}, AI-Generated: {(train_y==1).sum().item()})")
    print(f"    Val Samples  : {len(val_y):,} (Authentic: {(val_y==0).sum().item()}, AI-Generated: {(val_y==1).sum().item()})")
    print(f"    Test Samples : {len(test_y):,} (Authentic: {(test_y==0).sum().item()}, AI-Generated: {(test_y==1).sum().item()})")

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=args.batch_size, shuffle=False)

    # 2. Instantiate Model Head
    model = DINOv2ClassificationHead(
        in_features=EMBEDDING_DIM,
        hidden_dim=args.hidden_dim,
        num_classes=len(classes),
        dropout=args.dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Step 2] Model Architecture Instantiated:")
    print(f"    Input Features : {EMBEDDING_DIM}")
    print(f"    Hidden Dim     : {args.hidden_dim}")
    print(f"    Total Head Parameters : {total_params:,} ({total_params/1e3:.2f} k)")
    print(f"    Trainable Parameters  : {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 3. Training Loop
    print("\n[Step 3] Executing Training Loop...")
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    best_weights = None
    training_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(by)
            preds = logits.argmax(dim=-1)
            correct += (preds == by).sum().item()
            total += len(by)

        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total

        # Validation
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                loss = criterion(logits, by)
                v_loss += loss.item() * len(by)
                preds = logits.argmax(dim=-1)
                v_correct += (preds == by).sum().item()
                v_total += len(by)

        epoch_val_loss = v_loss / v_total
        epoch_val_acc = v_correct / v_total

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)

        print(f"  Epoch {epoch:02d}/{args.epochs:02d} | "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc*100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc*100:.2f}%", end="")

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_weights = model.state_dict().copy()
            print("  [BEST SAVED]")
        else:
            print()

    total_training_time = time.perf_counter() - training_start
    print(f"\nTraining completed in {total_training_time:.2f} seconds.")
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")

    # 4. Save Best Model
    os.makedirs(args.model_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.model_dir, "dinov2_authenticity_head.pth")
    checkpoint_payload = {
        "head_state_dict": best_weights,
        "classes": classes,
        "class_to_idx": {c: i for i, c in enumerate(classes)},
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dim": args.hidden_dim,
        "num_classes": len(classes),
        "best_val_acc": best_val_acc,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "training_time_sec": total_training_time,
        "date_trained": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    torch.save(checkpoint_payload, checkpoint_path)
    print(f"[Step 4] Saved trained classification head checkpoint to: {checkpoint_path}")

    # 5. Evaluate on Held-out Test Split
    print("\n[Step 5] Evaluating Best Model on Test Set...")
    model.load_state_dict(best_weights)
    model.eval()

    all_preds = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            logits = model(bx)
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Prob of AI-generated
            all_targets.extend(by.numpy())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)

    test_acc = accuracy_score(all_targets, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)
    roc_auc = roc_auc_score(all_targets, all_probs)
    cm = confusion_matrix(all_targets, all_preds)

    print("=" * 60)
    print("FINAL HELD-OUT TEST METRICS (DINOv2 Foundation Model):")
    print(f"  Accuracy  : {test_acc*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}% (Macro)")
    print(f"  Recall    : {rec*100:.2f}% (Macro)")
    print(f"  F1-Score  : {f1*100:.2f}% (Macro)")
    print(f"  ROC-AUC   : {roc_auc*100:.2f}%")
    print("  Confusion Matrix:")
    print(f"    [[TN={cm[0,0]}, FP={cm[0,1]}],")
    print(f"     [FN={cm[1,0]}, TP={cm[1,1]}]]")
    print("=" * 60)

    # 6. Save Test Outputs & Curves
    os.makedirs(args.output_dir, exist_ok=True)

    # Test report text
    report_text = f"""EXPLAINABLE AI IMAGE AUTHENTICITY VERIFICATION
DINOv2-SMALL (ViT-S/14) FROZEN BACKBONE + MLP HEAD
TEST EVALUATION REPORT (N={len(all_targets)})

Test Accuracy : {test_acc*100:.2f}%
Precision     : {prec*100:.2f}% (Macro)
Recall        : {rec*100:.2f}% (Macro)
F1-Score      : {f1*100:.2f}% (Macro)
ROC-AUC       : {roc_auc*100:.2f}%

Detailed Classification Breakdown:
{classification_report(all_targets, all_preds, target_names=classes, digits=4)}

Confusion Matrix:
[[{cm[0,0]:>5} {cm[0,1]:>5}]
 [{cm[1,0]:>5} {cm[1,1]:>5}]]
TN: {cm[0,0]}, FP: {cm[0,1]}, FN: {cm[1,0]}, TP: {cm[1,1]}
"""
    report_file = os.path.join(args.output_dir, "dinov2_test_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)

    # Confusion matrix CSV
    cm_csv_file = os.path.join(args.output_dir, "dinov2_confusion_matrix.csv")
    np.savetxt(cm_csv_file, cm, delimiter=",", fmt="%d", header="Pred_Authentic,Pred_AIGenerated", comments="")

    # Confusion matrix PNG
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("DINOv2 Test Set Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=20)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('Actual True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "dinov2_confusion_matrix.png"), dpi=200)
    plt.close()

    # Loss Curve PNG
    epochs_range = range(1, args.epochs + 1)
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs_range, train_losses, 'o-', label='Train Loss', color='tab:blue')
    plt.plot(epochs_range, val_losses, 's-', label='Validation Loss', color='tab:orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (CrossEntropy)')
    plt.title('DINOv2 Classifier Head - Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "dinov2_loss_curve.png"), dpi=200)
    plt.close()

    # Accuracy Curve PNG
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs_range, [a * 100 for a in train_accs], 'o-', label='Train Accuracy', color='tab:blue')
    plt.plot(epochs_range, [a * 100 for a in val_accs], 's-', label='Validation Accuracy', color='tab:green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('DINOv2 Classifier Head - Accuracy Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "dinov2_accuracy_curve.png"), dpi=200)
    plt.close()

    # Config JSON
    config_dict = {
        "model": "DINOv2-Small (ViT-S/14)",
        "in_features": EMBEDDING_DIM,
        "hidden_dim": args.hidden_dim,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "best_val_acc": float(best_val_acc),
        "test_acc": float(test_acc),
        "test_precision": float(prec),
        "test_recall": float(rec),
        "test_f1": float(f1),
        "test_roc_auc": float(roc_auc),
        "training_time_sec": float(total_training_time),
        "confusion_matrix": cm.tolist()
    }
    with open(os.path.join(args.output_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=4)

    return config_dict, (train_losses, val_losses, train_accs, val_accs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DINOv2 Classification Head on Cached Features")
    parser.add_argument("--cache_dir", type=str, default="outputs/features")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_classifier(args)
