from typing import OrderedDict
import torch
import torch.nn as nn
import os
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from torch.utils.data import Dataset, DataLoader
from models.resnet import generate_model
import csv
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import cc3d
from skimage import exposure
import torchio as tio

TRAIN_RESULTS = "1_No_transpose/Results/resnet10_med.csv"
BEST_MODEL_PATH = "1_No_transpose/Best_models/resnet10_med.pth"
PRETRAINED_PATH = "Pretrained-weights/resnet3d_10_med.pth"
TRAIN_CSV_PATH = '/mnt/storage/deva/Data/ultimate_train_bspline.csv'
TEST_CSV_PATH = '/mnt/storage/deva/Data/ultimate_test_bspline.csv'
VALIDATION_CSV_PATH = '/mnt/storage/deva/Data/ultimate_validation_bspline.csv'

n_classes = 1
n_input_channels = 1
model_depth = 10

# Function for removing the background
def background_removal(image):
    # Min-max normalizing the image
    image = (image-image.min())/(image.max()-image.min())
    # Thresholding the image
    threshold = 0.5
    labels_in = image > threshold
    # Getting the connected components from the thresholded image
    dusting_threshold = 1000
    connectivity=26
    labels_out = cc3d.dust(labels_in, threshold=dusting_threshold, connectivity=connectivity, in_place=False)
    x,y,z = labels_out.nonzero()
    return min(x), max(x), min(y), max(y)

def histogram_equalize_3d(volume):
    volume_flat = volume.flatten()
    equalized_flat = exposure.equalize_hist(volume_flat)  # Apply equalization
    return equalized_flat.reshape(volume.shape)  # Reshape back to 3D

class BreastCancer3DDataset(Dataset):
    def __init__(self, csv_path, target_shape=(96, 96, 96), transform=None):
        self.data = pd.read_csv(csv_path)
        self.data['BSplinePath'] = self.data['BSplinePath'].apply(lambda x: os.path.abspath(x))
        self.labels = self.data['output'].values
        self.image_paths = self.data['BSplinePath'].values
        self.window_center = self.data['WindowCenter'].values
        self.window_width = self.data['WindowWidth'].values
        self.target_shape = target_shape
        self.valid_indices = self.check_all_files()
        self.transform = transform
    def check_all_files(self):
        valid_indices = []
        for idx, image_path in enumerate(self.image_paths): 
            try:
                self.load_and_preprocess_image(image_path, self.window_center[idx], self.window_width[idx])
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
        # window_center = self.data.iloc[real_idx]['WindowCenter']
        # window_width = self.data.iloc[real_idx]['WindowWidth']
        image = self.load_and_preprocess_image(image_path, self.window_center[real_idx], self.window_width[real_idx])

        subject = tio.Subject(
                    image = tio.ScalarImage(tensor=torch.tensor(image, dtype=torch.float32), 
                    label=torch.tensor(label, dtype=torch.long)))

        if self.transform:
            subject = self.transform(subject)
        
        image_tensor = subject.image.data 
        label_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, label_tensor

    def load_and_preprocess_image(self, image_path, window_center, window_width):
        image = nib.load(image_path)
        #dtype = uint16
        if(image.header.get_data_dtype() == 'uint16'):
            image = image.get_fdata()
        #dtype = uint8 [('R', 'u1'), ('G', 'u1'), ('B', 'u1'), ('A', 'u1')]
        else:
            image = image.dataobj
            image = np.array(image[:,:,:,0,0])

        image = 255 * ((image - image.min())/(image.max()-image.min()))
        image = image.astype(np.uint8)
        x, y, z = image.shape
        # if (y==z):
        #     image = image.transpose((1,2,0))

        # window_min = window_center - (window_width / 2)
        # window_max = window_center + (window_width / 2)
        # image = np.clip(image, window_min, window_max)
        # image = (image - window_min) / (window_max - window_min)

        image = histogram_equalize_3d(image)
        image = (image - image.min())/(image.max() - image.min())
        min_x, max_x, min_y, max_y = background_removal(image)
        image = image[min_x:max_x,min_y:max_y,:]

        zoom_factors = [t / s for t, s in zip(self.target_shape, image.shape)]
        resized_image = zoom(image, zoom_factors, order=1)
        resized_image = np.interp(resized_image, (resized_image.min(), resized_image.max()), (0, 1))
        # **Stack grayscale image into 3 channels (Fake RGB)**
        stacked_image = np.stack([resized_image] * 1, axis=0)  # Shape: (3, H, W, D)
        mean = np.mean(stacked_image)
        std = np.std(stacked_image)
        stacked_image = (stacked_image - mean) / (std + 1e-8)
        # print(stacked_image.shape)

 
        return stacked_image  # Return a 3-channel image
        # return np.expand_dims(resized_image, axis=0)  # Add channel dim
    
def custom_collate_fn(batch):
    images, labels = zip(*batch)  # Unpack batch
    return torch.stack(images), torch.tensor(labels)

def train_model(train_loader, val_loader, metrics_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = generate_model(model_depth=model_depth, n_classes=n_classes, n_input_channels=n_input_channels)
    
    pretrained_weights = torch.load(PRETRAINED_PATH, map_location='cpu')
    # Load DataParallel model checkpoint

    # Create new OrderedDict without "module." prefix
    new_state_dict = OrderedDict()
    for k, v in pretrained_weights["state_dict"].items():  # Adjust key if stored differently
        new_key = k.replace("module.", "")  # Remove "module." prefix
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict, strict=False)

    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5, verbose=True)
    save_path = BEST_MODEL_PATH
    num_epochs = 100
    early_stop_counter = 0
    patience = 15
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for batch in train_loader:
            inputs, labels = batch[:2]
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()* inputs.size(0)
            probabilities = torch.sigmoid(outputs)
            preds = (probabilities > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        train_acc = train_correct / train_total if train_total > 0 else 0
        train_loss /= train_total
        # Validation step
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_labels, all_outputs = [], []
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch[:2]
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()* inputs.size(0)
                probabilities = torch.sigmoid(outputs)
                preds = (probabilities > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
            
                all_labels.extend(labels.cpu().numpy())
                all_outputs.extend(outputs.cpu().numpy())
        val_acc = val_correct / val_total if val_total > 0 else 0
        val_loss /= val_total
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Learning Rate: {current_lr}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        if early_stop_counter >= patience:
            print("Early stopping triggered")
            break

        metrics["epoch"].append(epoch + 1)
        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["train_accuracy"].append(train_acc)
        metrics["val_accuracy"].append(val_acc)
        metrics["learning_rate"].append(current_lr)

    with open(metrics_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=metrics.keys())
        writer.writeheader()
        for i in range(len(metrics['epoch'])):
            writer.writerow({key: metrics[key][i] for key in metrics})
    print("metrics are saved")

if __name__ == "__main__":
    metrics_file = TRAIN_RESULTS
    metrics = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "learning_rate": []
    }


    transform = tio.Compose([
    tio.Pad((10, 10, 10)),
    # tio.RandomFlip(axes=('LR',), p=0.5),  # horizontal flip (left-right)
    # tio.RandomFlip(axes=('AP',), p=0.5),  # vertical flip (anterior-posterior)
    tio.RandomNoise(mean=0, std=0.02, p=0.25),  # Gaussian noise with std=0.02
    tio.RandomAffine(
        degrees=20,  # random rotation up to ±20° in all axes
        translation=0,  # no translation
        scales=1.0,     # no scaling
        isotropic=True, 
        default_pad_value='minimum',
        p=0.5  # apply with 75% probability
    ),
    tio.CropOrPad((96, 96, 96))
    ])

    train_dataset = BreastCancer3DDataset(TRAIN_CSV_PATH, transform=transform)
    val_dataset = BreastCancer3DDataset(VALIDATION_CSV_PATH)
    test_dataset = BreastCancer3DDataset(TEST_CSV_PATH)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)


    train_model(train_loader, val_loader, metrics_file)
