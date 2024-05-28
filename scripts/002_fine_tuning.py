import sys
import numpy as np
import timm
import torch
from torch import tensor
import torch.nn as nn
from torchvision.transforms import InterpolationMode, transforms
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from PIL import Image
import os
from tqdm import tqdm
import getpass
import socket
from datetime import datetime
from sklearn.metrics import precision_score, recall_score

# Set the PyTorch device (GPU/cuda or CPU)
if torch.cuda.is_available():
    dev = "cuda"
    device = torch.device(dev)

    gpu_name = torch.cuda.get_device_name(torch.device("cuda"))
    print(f"GPU name: {gpu_name} ({torch.cuda.device_count()} available)")
    
    print("Host name: ", socket.gethostname())  # Retrieve the hostname of the current system to determine the environment
    print("User name: ", getpass.getuser())  # Retrieve the current user's username

    # If the notebook is running on the JASMIN GPU cluster, select the GPU with the most free memory
    if socket.gethostname() == "gpuhost001.jc.rl.ac.uk":

        def select_gpu_with_most_free_memory():
            max_memory_available = 0
            gpu_id_with_max_memory = 0
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_device(i)
                free_mem, total_mem = torch.cuda.mem_get_info(i)
                free_mem_gib = free_mem / (1024 ** 3)
                free_mem_rounded = round(free_mem_gib, 2)
                print(f"GPU {i} free memory: {free_mem_rounded} GiB")
                if free_mem_gib >= max_memory_available:  # >= biases away from GPU 0, which most JASMIN users default to
                    max_memory_available = free_mem_gib
                    gpu_id_with_max_memory = i
            return gpu_id_with_max_memory

        best_gpu = select_gpu_with_most_free_memory()

        torch.cuda.set_device(best_gpu)
        print(f"Using GPU: {best_gpu}")
    
    else:
        _, max_memory = torch.cuda.mem_get_info()
        max_memory = max_memory / (1024 ** 3)
        print(f"GPU memory: {max_memory} GiB")

else:
    dev = "cpu"
    device = torch.device(dev)
    print("No GPU available.")

gpu_override = False
if gpu_override:
    torch.cuda.set_device(3)
    print(f"OVERRIDE: Using GPU: {3}")

CROP_SIZE = 182
BACKBONE = "vit_large_patch14_dinov2"
weight_path = "../models/deepfaune-vit_large_patch14_dinov2.lvd142m.pt"

jasmin = True

if jasmin:
    train_path = "../data/split_data/train"
    val_path = "../data/split_data/val"
    test_path = "../data/split_data/test"
else:
    train_path = "/media/tom-ratsakatika/CRUCIAL 4TB/FCC Camera Trap Data/split_data/train"
    val_path = "/media/tom-ratsakatika/CRUCIAL 4TB/FCC Camera Trap Data/split_data/val"
    test_path = "/media/tom-ratsakatika/CRUCIAL 4TB/FCC Camera Trap Data/split_data/test"

ANIMAL_CLASSES = ["badger", "ibex", "red deer", "chamois", "cat", "goat", "roe deer", "dog", "squirrel", "equid", "genet",
                  "hedgehog", "lagomorph", "wolf", "lynx", "marmot", "micromammal", "mouflon",
                  "sheep", "mustelid", "bird", "bear", "nutria", "fox", "wild boar", "cow"]

class AnimalDataset(Dataset):
    def __init__(self, directory, transform=None, preload_to_gpu=False):
        self.directory = directory
        self.transform = transform
        self.images = []
        self.labels = []
        self.preload_to_gpu = preload_to_gpu

        for label in os.listdir(directory):
            label_dir = os.path.join(directory, label)
            if os.path.isdir(label_dir):
                for image in os.listdir(label_dir):
                    image_path = os.path.join(label_dir, image)
                    self.images.append(image_path)
                    self.labels.append(ANIMAL_CLASSES.index(label))

        if self.preload_to_gpu:
            self.preload_images()

    def preload_images(self):
        self.loaded_images = []
        for image_path in tqdm(self.images, desc="Preloading images to GPU"):
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            self.loaded_images.append(image.to(device))
        self.labels = torch.tensor(self.labels, device=device)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.preload_to_gpu:
            return self.loaded_images[idx], self.labels[idx]
        else:
            image_path = self.images[idx]
            label = self.labels[idx]
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label

class Classifier(nn.Module):
    def __init__(self, freeze_up_to_layer=16):
        super(Classifier, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = timm.create_model(BACKBONE, pretrained=False, num_classes=len(ANIMAL_CLASSES), dynamic_img_size=True)
        state_dict = torch.load(weight_path, map_location=torch.device(device))['state_dict']
        self.model.load_state_dict({k.replace('base_model.', ''): v for k, v in state_dict.items()})

        # Freeze layers up to the specified layer
        if freeze_up_to_layer is not None:
            for name, param in self.model.named_parameters():
                if self._should_freeze_layer(name, freeze_up_to_layer):
                    param.requires_grad = False

        self.transforms = transforms.Compose([
            transforms.Resize(size=(CROP_SIZE, CROP_SIZE), interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=None),
            transforms.ToTensor(),
            transforms.Normalize(mean=tensor([0.4850, 0.4560, 0.4060]), std=tensor([0.2290, 0.2240, 0.2250]))
        ])

    def _should_freeze_layer(self, name, freeze_up_to_layer):
        if 'blocks' in name:
            block_num = int(name.split('.')[1])
            if block_num <= freeze_up_to_layer:
                return True
        return False

    def forward(self, x):
        return self.model(x)

    def predict(self, image):
        img_tensor = self.transforms(image).unsqueeze(0)
        with torch.no_grad():
            output = self.forward(img_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            top_p, top_class = probabilities.topk(1, dim=1)
            return ANIMAL_CLASSES[top_class.item()], top_p.item()

# Custom loss function with a higher penalty for misclassifying wild boar
class CustomLoss(nn.Module):
    def __init__(self, penalty_weight=0.0):
        super(CustomLoss, self).__init__()
        self.penalty_weight = penalty_weight
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        loss = self.ce_loss(outputs, targets)
        wild_boar_index = ANIMAL_CLASSES.index("wild boar")
        wild_boar_mask = (targets == wild_boar_index)
        if wild_boar_mask.sum() > 0:
            wild_boar_loss = self.ce_loss(outputs[wild_boar_mask], targets[wild_boar_mask])
            loss += self.penalty_weight * wild_boar_loss
        return loss

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    accuracy = 100 * correct / total
    overall_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    overall_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Calculate precision and recall for wild boar
    wild_boar_index = ANIMAL_CLASSES.index("wild boar")
    wild_boar_precision = precision_score(all_labels, all_preds, labels=[wild_boar_index], average='macro', zero_division=0)
    wild_boar_recall = recall_score(all_labels, all_preds, labels=[wild_boar_index], average='macro', zero_division=0)
    
    return running_loss / len(dataloader), accuracy, overall_precision, overall_recall, wild_boar_precision, wild_boar_recall

def test(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def save_model(model, total_epochs, learning_rate, now, penalty_weight):
    model_save_path = f"../models/{now}-deepfaune-finetuned-epochs-{total_epochs}-lr-{learning_rate}-wbpenalty-{penalty_weight}.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

def main():
    initial_epochs = 5  # Set the number of epochs
    batch_size = 32  # Set the batch size
    learning_rate = 1e-5  # Reduced learning rate for fine-tuning
    total_epochs = initial_epochs
    penalty_weight = 0.0  # Initial penalty weight for wild boar class

    transform = transforms.Compose([
        transforms.Resize((CROP_SIZE, CROP_SIZE), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=tensor([0.4850, 0.4560, 0.4060]), std=tensor([0.2290, 0.2240, 0.2250]))
    ])

    print('Loading training data...')
    train_dataset = AnimalDataset(train_path, transform=transform, preload_to_gpu=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print('Loading validation data...')
    val_dataset = AnimalDataset(val_path, transform=transform, preload_to_gpu=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = Classifier(freeze_up_to_layer=16).to(device)  # Freeze up to the 16th layer

    criterion = CustomLoss(penalty_weight=penalty_weight)  # Custom loss with initial penalty weight
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize best_val_loss
    best_val_loss = float('inf')

    # Evaluate validation set before training
    print('Initial validation evaluation...')
    val_loss, val_accuracy, val_precision, val_recall, wb_precision, wb_recall = validate(model, val_loader, criterion, device)
    print(f'Initial Validation Loss: {val_loss}, Initial Validation Accuracy: {val_accuracy}%')
    print(f'Overall Precision: {val_precision}, Overall Recall: {val_recall}')
    print(f'Wild Boar Precision: {wb_precision}, Wild Boar Recall: {wb_recall}')

    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    print('Training started...')
    for epoch in range(initial_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy, val_precision, val_recall, wb_precision, wb_recall = validate(model, val_loader, criterion, device)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}%')
        print(f'Overall Precision: {val_precision}, Overall Recall: {val_recall}')
        print(f'Wild Boar Precision: {wb_precision}, Wild Boar Recall: {wb_recall}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("Saving best epoch...")
            save_model(model, total_epochs, learning_rate, now, penalty_weight)

    if val_loss != best_val_loss:
        now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        print("Saving model's current state...")
        save_model(model, total_epochs, learning_rate, now, penalty_weight)

    # Option to continue training
    while True:
        more_epochs = int(input("Enter the number of additional epochs to continue training (0 to stop): "))
        if more_epochs == 0:
            break
        while True:
            learning_rate = float(input("Enter the learning rate for the additional epochs (default 1e-5): "))
            if learning_rate <= 1e-4:
                break
            else:
                print("Learning rate too high")
        penalty_weight = float(input("Enter the penalty weight for wild boar class (default 0): "))
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Update optimizer with new learning rate
        criterion = CustomLoss(penalty_weight=penalty_weight)  # Update criterion with new penalty weight
        total_epochs += more_epochs
        now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        
        for epoch in range(more_epochs):
            train_loss = train(model, train_loader, criterion, optimizer, device)
            val_loss, val_accuracy, val_precision, val_recall, wb_precision, wb_recall = validate(model, val_loader, criterion, device)
            print(f'Epoch {total_epochs - more_epochs + epoch + 1}, Train Loss: {train_loss}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}%')
            print(f'Overall Precision: {val_precision}, Overall Recall: {val_recall}')
            print(f'Wild Boar Precision: {wb_precision}, Wild Boar Recall: {wb_recall}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("Saving best epoch...")
                save_model(model, total_epochs, learning_rate, now, penalty_weight)
        
        if val_loss != best_val_loss:
            now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            print("Saving model's current state...")
            save_model(model, total_epochs, learning_rate, now, penalty_weight)

    # Load test data
    print('Loading test data...')
    test_dataset = AnimalDataset(test_path, transform=transform, preload_to_gpu=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Test the model
    print('Testing the model...')
    test_accuracy = test(model, test_loader, device)
    print(f'Test Accuracy: {test_accuracy}%')

    # Return critical variables for further experimentation
    return model, train_loader, val_loader, test_loader, criterion, optimizer, total_epochs

if __name__ == '__main__':
    model, train_loader, val_loader, test_loader, criterion, optimizer, total_epochs = main()