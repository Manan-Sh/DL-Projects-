"""
This model trains and tests data using the ViT model and classifies data. By default this model has been used for fruit classification. For future use of this model just change the traing and testing data accordingly to see the accuracies.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from vit_pytorch import ViT

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define hyperparameters
batch_size = 64
epochs = 10
learning_rate = 0.001
weight_decay = 1e-4

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize images to 32x32 pixels
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load datasets
dataset_path = '/Users/manansharma/Desktop/fruits-360/Training'
test_dataset_path = '/Users/manansharma/Desktop/fruits-360/Test'  # Path to test data

train_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.ImageFolder(root=test_dataset_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the Vision Transformer model with updated hyperparameters
model = ViT(
    image_size=32,
    patch_size=4,
    num_classes=len(train_dataset.classes),
    dim=512,
    depth=8,
    heads=12,
    mlp_dim=1024,
    dropout=0.05,
    emb_dropout=0.05
).to(device)

# Define loss function and optimizer with weight decay
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Training loop
for epoch in range(epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Track loss
        running_loss += loss.item()

        # Track accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if i % 100 == 99:
            accuracy = 100 * correct / total
            print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}], Loss: {running_loss / 100:.4f}, Accuracy: {accuracy:.2f}%')
            running_loss = 0.0

print('Finished Training')

# Evaluation loop
model.eval()  # Set model to evaluation mode
correct = 0
total = 0
with torch.no_grad():  # No need to track gradients for evaluation
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f'Test Accuracy: {test_accuracy:.2f}%')
