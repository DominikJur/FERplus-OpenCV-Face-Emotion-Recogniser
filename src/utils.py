import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm


CURRENT_BEST_MODEL_PATH = "models\\ResNet_epoch_66.pth"

def get_model(num_classes, device, path=CURRENT_BEST_MODEL_PATH):
    """
    Load a pre-trained model and modify the final layer to match the number of classes.
    """
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    model.load_state_dict(
        torch.load(CURRENT_BEST_MODEL_PATH, map_location=torch.device(device))
    )
    print("Model loaded successfully")
    model.eval()
    
    return model

def train_model(
    model,
    train_loader,
    criterion,
    optimizer,
    num_epochs=5,
    device="gpu" if torch.cuda.is_available() else "cpu",
):
    model.train()

    model.to(device)

    for epoch in range(num_epochs):
        running_loss = 0.0
        start_time = time.time()  # Start timing epoch
        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{num_epochs}",
        )
        for i, (images, labels) in progress_bar:
            images, labels = images.to(device), labels.to(
                device
            )  # Move data to GPU if available

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                progress_bar.set_postfix(current_loss=loss.item())
                running_loss = 0.0

        epoch_time = time.time() - start_time
        print(
            f"Epoch {epoch + 1}/{num_epochs} completed in {epoch_time // 60:.0f} minutes and {epoch_time % 60:.0f} seconds"
        )

        torch.save(model.state_dict(), f"..\\models\\{type(model).__name__}_epoch_{epoch+1}.pth")


def evaluate_model(
    model, test_loader, device="gpu" if torch.cuda.is_available() else "cpu"
):
    model.eval()  # turn train mode OFF

    total = 0
    correct = 0

    with torch.no_grad():  # No need to compute gradients
        eval_bar = tqdm(test_loader, desc="Evaluating")
        for images, labels in eval_bar:
            images, labels = images.to(device), labels.to(
                device
            )  # Move data to GPU if available
            outputs = model(images)
            _, predicted = torch.max(
                outputs, 1
            )  # Get the index of the max log-probability
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            current_acc = 100 * correct / total
            eval_bar.set_postfix(current_acc=f"{current_acc:.2f}%")

    print(f"Final Accuracy: {100 * correct / total:.2f}%")


def imshow(img, cmap="gray"):
    npimg = img[0].numpy() if isinstance(img, tuple) else img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap=cmap)
    plt.show()


def display_model_predictions(
    model, test_loader, num_images=5, cmap="gray", classes_dict=None
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    if classes_dict is not None:
        labels_disp = [classes_dict[l.item()] for l in labels[:num_images]]
        predicted_disp = [classes_dict[p.item()] for p in predicted[:num_images]]
    else:
        labels_disp = [l.item() for l in labels[:num_images]]
        predicted_disp = [p.item() for p in predicted[:num_images]]

    rows = int(np.ceil(num_images / 5))
    columns = 5
    _, axes = plt.subplots(rows, columns, figsize=(12, 6))

    for i in range(rows * columns):
        if i < num_images:
            current_row = i // 5
            image = images[i].cpu() / 2 + 0.5  # Unnormalize and move to cpu
            if rows > 1:
                axes[current_row, i % 5].imshow(
                    image.permute(1, 2, 0).numpy(), cmap=cmap
                )
                axes[current_row, i % 5].set_title(
                    f"Label: {labels_disp[i]},\nPredicted: {predicted_disp[i]}"
                )
                axes[current_row, i % 5].axis("off")
            else:
                axes[i].imshow(image.permute(1, 2, 0).numpy(), cmap=cmap)
                axes[i].set_title(
                    f"Label: {labels_disp[i]},\nPredicted: {predicted_disp[i]}"
                )
                axes[i].axis("off")
        else:
            if rows > 1:
                axes[current_row, i % 5].axis("off")
            else:
                axes[i].axis("off")
    plt.tight_layout()
    plt.show()
