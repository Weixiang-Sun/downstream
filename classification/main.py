import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader, random_split, Subset
import argparse
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import Counter
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Train a classification model")
parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet50', 'densenet121', 'densenet169'], help='Model to use for training')
parser.add_argument('--dataset_path', type=str, default='./dataset', help='Path to the dataset')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
parser.add_argument('--test_per_epoch', type=int, default=1, help='Frequency of testing per epoch')
args = parser.parse_args()

dataset_path = args.dataset_path
batch_size = args.batch_size
epochs = args.epochs
test_per_epoch = args.test_per_epoch
learning_rate = 0.001

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

num_classes = len(dataset.classes)

# print("Claculating dataset distribution, this part maybe a little slow.")
# train_class_counts = np.bincount([label for _, label in train_dataset], minlength=num_classes)
# test_class_counts = np.bincount([label for _, label in test_dataset], minlength=num_classes)

# print("Training set class distribution:")
# for class_idx, count in enumerate(train_class_counts):
#     print(f"{dataset.classes[class_idx]}: {count}")

# print("\nTesting set class distribution:")
# for class_idx, count in enumerate(test_class_counts):
#     print(f"{dataset.classes[class_idx]}: {count}")

if args.model == 'resnet18':
    model = models.resnet18(weights=None)
elif args.model == 'resnet50':
    model = models.resnet50(weights=None)
elif args.model == 'densenet121':
    model = models.densenet121(weights=None)
elif args.model == 'densenet169':
    model = models.densenet169(weights=None)

if 'resnet' in args.model:
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
else:
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, num_classes)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

results_file = "test_results.txt"
if os.path.exists(results_file):
    os.remove(results_file)

def evaluate(epoch):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    accuracy = 100. * (all_labels == all_preds).sum() / len(all_labels)

    if num_classes == 2:
        roc_auc = roc_auc_score(all_labels, all_preds)
        precision_vals, recall_vals, _ = precision_recall_curve(all_labels, all_preds, pos_label=1)
        pr_auc = auc(recall_vals, precision_vals)

        with open(results_file, "a") as f:
            f.write(f"Epoch: {epoch+1}")
            f.write(f"Accuracy: {accuracy:.2f}%")
            f.write(f"Precision: {precision:.4f}")
            f.write(f"Recall: {recall:.4f}")
            f.write(f"F1 Score: {f1:.4f}")
            f.write(f"Confusion Matrix:{conf_matrix}")
        print(f"Epoch: {epoch+1}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Confusion Matrix:{conf_matrix}")

        plt.figure()
        plt.plot(recall_vals, precision_vals, marker='.', label='Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.savefig(f'./precision_recall_curve_epoch_{epoch+1}.png')
        plt.show()
    else:
        with open(results_file, "a") as f:
            f.write(f"Epoch: {epoch+1}")
            f.write(f"Accuracy: {accuracy:.2f}%")
            f.write(f"Precision: {precision:.4f}")
            f.write(f"Recall: {recall:.4f}")
            f.write(f"F1 Score: {f1:.4f}")
            f.write(f"Confusion Matrix:{conf_matrix}")
        print(f"Epoch: {epoch+1}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Confusion Matrix:{conf_matrix}")
        # draw Confusion Matrix
        data = conf_matrix
        fig, ax = plt.subplots()
        cax = ax.matshow(data, cmap='viridis')
        for (i, j), val in np.ndenumerate(data):
            ax.text(j, i, f'{val}', ha='center', va='center', color='white')
        fig.colorbar(cax)
        plt.savefig(f'./confusion_matrix_epoch.png')
        plt.show()


        wandb.log({"test_accuracy": accuracy, "test_precision": precision, "test_recall": recall, "test_f1": f1, "test_confusion_matrix": wandb.Image(plt)})


def train():
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        with tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch [{epoch+1}/{epochs}]")
            for inputs, labels in tepoch:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                tepoch.set_postfix(loss=running_loss / (total // batch_size), accuracy=100. * correct / total)
                wandb.log({"train_loss": running_loss / (total // batch_size), "train_accuracy": 100. * correct / total})

        if (epoch + 1) % test_per_epoch == 0:
            evaluate(epoch)

if __name__ == "__main__":
    wandb.login(key='77523b1204d462f1b95f4f3edf1efe30aa8be263')
    wandb.init(project = 'cls-train', config = args)
    train()