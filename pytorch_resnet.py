import os
import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
import seaborn as sns
from sklearn.utils import resample
from models.resnet import generate_model
import csv
from sklearn.metrics import precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt
# Initialize a list to store epoch metrics
metrics = []
skipped_files_count = []

# Define the dataset class
class BreastCancer3DDataset(Dataset):
    def __init__(self, csv_path, target_shape=(96, 96, 96), transform=None):
        self.data = pd.read_csv(csv_path)
        self.data['NIFTI location'] = self.data['NIFTI location'].apply(lambda x: os.path.abspath(x))
        self.labels = self.data['label'].values
        self.image_paths = self.data['NIFTI location'].values
        self.target_shape = target_shape
        self.transform = transform
        self.valid_indices = self.check_all_files()
        # self.valid_indices, self.skipped_count = self.filter_valid_files()
        # skipped_files_count.append({"csv_path": csv_path, "skipped_files": self.skipped_count})
    def check_all_files(self):
        valid_indices = []
        for idx, image_path in enumerate(self.image_paths): 
            try:
                self.load_and_preprocess_image(image_path)
                valid_indices.append(idx)
            except Exception as e:
                print(f"Warning: File {image_path} caused an error {e} and will be skipped")
        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        image_path = self.image_paths[real_idx]
        label = self.labels[real_idx]
        image = self.load_and_preprocess_image(image_path)
        if self.transform:
            image = self.transform(image)
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def load_and_preprocess_image(self, image_path):
        image = nib.load(image_path)
        if(image.header.get_data_dtype() == 'uint16'):
            image = image.get_fdata()
        else: #[('R', 'u1'), ('G', 'u1'), ('B', 'u1'), ('A', 'u1')]
            image = image.dataobj
            image = np.array(image)['R']
        image = 255 * ((image - image.min())/(image.max()-image.min()))
        image = image.astype(np.uint8)
        zoom_factors = [t / s for t, s in zip(self.target_shape, image.shape)]
        resized_image = zoom(image, zoom_factors, order=1)
        resized_image = np.interp(resized_image, (resized_image.min(), resized_image.max()), (0, 1))
        return np.expand_dims(resized_image, axis=0)  # Add channel dim

# Helper function to filter out skipped files
# def clean_loader(loader):
#     return [(images, labels) for images, labels in loader if images is not None]
def custom_collate_fn(batch):
    images, labels = zip(*batch)  # Unpack batch
    return torch.stack(images), torch.tensor(labels)

train_csv_path = '/mnt/storage/deva/Data/new_train_data.csv'
val_csv_path = '/mnt/storage/deva/Data/new_val_data.csv'
test_csv_path = '/mnt/storage/deva/Data/combined_test_data.csv'

# Create datasets and dataloaders
train_dataset = BreastCancer3DDataset(train_csv_path)
val_dataset = BreastCancer3DDataset(val_csv_path)
test_dataset = BreastCancer3DDataset(test_csv_path)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)

# Initialize the ResNet3D model
model = generate_model(model_depth=18, n_classes=2, n_input_channels=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Define bootstrap function
def bootstrap_auc(labels, probs, n_iterations=1000, ci=95):
    auc_scores = []
    for _ in range(n_iterations):
        indices = resample(range(len(labels)), n_samples=len(labels))
        resampled_labels = labels[indices]
        resampled_probs = probs[indices]
        auc_scores.append(roc_auc_score(resampled_labels, resampled_probs))
    lower = np.percentile(auc_scores, (100 - ci) / 2)
    upper = np.percentile(auc_scores, 100 - (100 - ci) / 2)
    return lower, upper

# Function to evaluate model
def evaluate_model(loader, model):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

# Training loop
num_epochs = 20
best_val_acc = 0.0
checkpoint_interval = 5

for epoch in range(1, num_epochs+1):
    # Training
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for batch in train_loader:
        images, labels = batch[:2]  # Unpacking is safe now
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc = 100 * correct / total
    
    #validation

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    for batch in val_loader:
        images, labels = batch[:2]  # Unpacking is safe now
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        val_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    val_acc = 100 * correct / total


    # Collect metrics
    epoch_metrics = {
        "epoch": epoch,
        "train_loss": train_loss / len(train_loader),
        "train_accuracy": train_acc,
        "validation_loss": val_loss / len(train_loader),
        "validation_accuracy": val_acc,
    }
    metrics.append(epoch_metrics)

    print(
        f"Epoch {epoch}/{num_epochs}, "
        f"Train Loss: {train_loss / len(train_loader):.4f}, Train Acc: {train_acc:.2f}%\n"
        f"Validation Loss: {val_loss / len(train_loader):.4f}, Validation Acc: {val_acc:.4f}, "
    )


    # Save model periodically and based on best validation AUC
    if epoch % checkpoint_interval == 0:
        checkpoint_path = f"checkpoint_epoch_{epoch + 1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch + 1}.")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_resnet3d_model.pth")
        print("Best model saved.")

# Save metrics to CSV after training
csv_file_path = "training_metrics_jan19.csv"
pd.DataFrame(metrics).to_csv(csv_file_path, index=False)
print(f"Metrics saved to {csv_file_path}")

# # Save skipped files count to CSV
# skipped_files_csv_path = "skipped_files_count_jan19.csv"
# pd.DataFrame(skipped_files_count).to_csv(skipped_files_csv_path, index=False)
# print(f"Skipped files count saved to {skipped_files_csv_path}")

# Test the model
model.load_state_dict(torch.load("best_resnet3d_model.pth"))
# test_labels, test_preds, test_probs = evaluate_model(test_loader, model)

# Load the best model
model.load_state_dict(torch.load("best_resnet3d_model.pth"))
model.eval()

# Evaluate on the test set
all_labels, all_preds, all_probs = [], [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# Convert to numpy arrays
all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

# Calculate confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate precision, recall, TPR, FPR
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
fpr, tpr, _ = roc_curve(all_labels, all_probs)

# Plot Confusion Matrix
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-cancer", "Cancer"],
            yticklabels=["Non-cancer", "Cancer"])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# Plot ROC Curve
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# # Print metrics
# print(f"Precision: {precision:.4f}")
# print(f"Recall: {recall:.4f}")
# print(f"AUC: {roc_auc:.4f}")


# # Metrics
# print(classification_report(test_labels, test_preds))
# sns.heatmap(confusion_matrix(test_labels, test_preds), annot=True, fmt="d", cmap="Blues")
# plt.show()
# fpr, tpr, _ = roc_curve(test_labels, test_probs)
# plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(test_labels, test_probs):.4f}")
# plt.legend()
# plt.show()

# # Metrics
# conf_matrix = confusion_matrix(test_labels, test_preds)
# print("Confusion Matrix:")
# print(conf_matrix)

# sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-cancer", "Cancer"], yticklabels=["Non-cancer", "Cancer"])
# plt.xlabel("Predicted Labels")
# plt.ylabel("True Labels")
# plt.title("Confusion Matrix")
# plt.show()
