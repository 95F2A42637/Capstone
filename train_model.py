import os
import copy
import torch
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
from torch import nn, optim

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. SETTINGS
# ============================================================

TRAIN_DIR = r"dataset\train"
VAL_DIR = r"dataset\val"
TEST_DIR = r"dataset\test"

MODEL_DIR = r"models"
OUTPUT_DIR = r"outputs"

BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 0.0001

IMAGE_SIZE = 224

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("Explainable AI Image Authenticity Verification")
print("=" * 60)
print(f"Device: {device}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print("=" * 60)


# ============================================================
# 3. IMAGE TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 4. LOAD DATASET
# ============================================================

print("\nLoading datasets...")

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_test_transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=val_test_transform
)

print(f"Training images   : {len(train_dataset)}")
print(f"Validation images : {len(val_dataset)}")
print(f"Testing images    : {len(test_dataset)}")

print("\nClass mapping:")
print(train_dataset.class_to_idx)

class_names = train_dataset.classes

print(f"Classes: {class_names}")


# ============================================================
# 5. DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 6. LOAD PRETRAINED RESNET18
# ============================================================

print("\nLoading pretrained ResNet18...")

weights = ResNet18_Weights.DEFAULT

model = resnet18(weights=weights)


# ============================================================
# 7. FREEZE BACKBONE
# ============================================================

for parameter in model.parameters():
    parameter.requires_grad = False

# Enable final convolutional block for learning
for parameter in model.layer4.parameters():
    parameter.requires_grad = True


# ============================================================
# 8. REPLACE CLASSIFIER
# ============================================================

number_of_features = model.fc.in_features

model.fc = nn.Linear(
    number_of_features,
    len(class_names)
)

model = model.to(device)


# ============================================================
# 9. LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

trainable_parameters = [
    parameter
    for parameter in model.parameters()
    if parameter.requires_grad
]

optimizer = optim.AdamW(
    trainable_parameters,
    lr=LEARNING_RATE
)


# ============================================================
# 10. TRAINING FUNCTION
# ============================================================

def train_one_epoch(model, loader):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_index, (images, labels) in enumerate(loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

        if (batch_index + 1) % 100 == 0:
            print(
                f"  Batch {batch_index + 1}/{len(loader)} "
                f"- Loss: {loss.item():.4f}"
            )

    epoch_loss = running_loss / len(loader)

    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# 11. VALIDATION FUNCTION
# ============================================================

def validate(model, loader):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(loader)

    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# 12. TRAIN MODEL
# ============================================================

print("\nStarting training...")
print("=" * 60)

best_val_accuracy = 0.0
best_model_weights = copy.deepcopy(model.state_dict())

train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []


for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print("-" * 50)

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader
    )

    val_loss, val_accuracy = validate(
        model,
        val_loader
    )

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    print(f"\nTrain Loss     : {train_loss:.4f}")
    print(f"Train Accuracy : {train_accuracy * 100:.2f}%")

    print(f"Val Loss       : {val_loss:.4f}")
    print(f"Val Accuracy   : {val_accuracy * 100:.2f}%")

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        best_model_weights = copy.deepcopy(
            model.state_dict()
        )

        print(">>> Best model updated.")


# ============================================================
# 13. RESTORE BEST MODEL
# ============================================================

model.load_state_dict(best_model_weights)

print("\nTraining completed.")
print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)


# ============================================================
# 14. SAVE MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "resnet18_authenticity.pth"
)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "class_to_idx": train_dataset.class_to_idx
    },
    model_path
)

print(f"\nModel saved to:")
print(model_path)


# ============================================================
# 15. TEST MODEL
# ============================================================

print("\nEvaluating on test dataset...")

model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        _, predictions = torch.max(outputs, 1)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


# ============================================================
# 16. TEST ACCURACY
# ============================================================

test_accuracy = accuracy_score(
    all_labels,
    all_predictions
)

print(
    f"\nTest Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# 17. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    all_labels,
    all_predictions,
    target_names=class_names,
    zero_division=0
)

print("\nClassification Report:")
print(report)

report_path = os.path.join(
    OUTPUT_DIR,
    "test_report.txt"
)

with open(report_path, "w") as file:

    file.write(
        "Explainable AI Image Authenticity Verification\n"
    )

    file.write("=" * 60 + "\n\n")

    file.write(
        f"Test Accuracy: {test_accuracy * 100:.2f}%\n\n"
    )

    file.write(
        "Classification Report\n"
    )

    file.write("-" * 60 + "\n")

    file.write(report)


# ============================================================
# 18. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

plt.figure(figsize=(7, 6))

plt.imshow(cm)

plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.xticks(
    range(len(class_names)),
    class_names,
    rotation=20
)

plt.yticks(
    range(len(class_names)),
    class_names
)

for i in range(len(class_names)):

    for j in range(len(class_names)):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.tight_layout()

confusion_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    confusion_path,
    dpi=200
)

plt.close()


# ============================================================
# 19. TRAINING CURVES
# ============================================================

epochs_range = range(
    1,
    EPOCHS + 1
)

plt.figure(figsize=(8, 5))

plt.plot(
    epochs_range,
    train_accuracies,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    epochs_range,
    val_accuracies,
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title("Training and Validation Accuracy")

plt.legend()

plt.grid(True)

plt.tight_layout()

accuracy_curve_path = os.path.join(
    OUTPUT_DIR,
    "accuracy_curve.png"
)

plt.savefig(
    accuracy_curve_path,
    dpi=200
)

plt.close()


# ============================================================
# 20. LOSS CURVE
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    epochs_range,
    train_losses,
    marker="o",
    label="Training Loss"
)

plt.plot(
    epochs_range,
    val_losses,
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Training and Validation Loss")

plt.legend()

plt.grid(True)

plt.tight_layout()

loss_curve_path = os.path.join(
    OUTPUT_DIR,
    "loss_curve.png"
)

plt.savefig(
    loss_curve_path,
    dpi=200
)

plt.close()


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PROJECT TRAINING SUMMARY")
print("=" * 60)

print(f"Training images   : {len(train_dataset)}")
print(f"Validation images : {len(val_dataset)}")
print(f"Testing images    : {len(test_dataset)}")

print(
    f"Best Val Accuracy : "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Test Accuracy     : "
    f"{test_accuracy * 100:.2f}%"
)

print("\nGenerated files:")

print(
    f"Model             : {model_path}"
)

print(
    f"Test Report       : {report_path}"
)

print(
    f"Confusion Matrix  : {confusion_path}"
)

print(
    f"Accuracy Curve    : {accuracy_curve_path}"
)

print(
    f"Loss Curve        : {loss_curve_path}"
)

print("\nTraining pipeline completed successfully.")
print("=" * 60)