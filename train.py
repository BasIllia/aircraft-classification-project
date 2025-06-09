# train.py

import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def main():
    parser = argparse.ArgumentParser(description="Train ResNet50 on aircraft dataset")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory with train/ (and val/) subfolders")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save models and class_names.json")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--save_epochs", type=int, nargs="*", default=[10],
                        help="Epoch numbers at which to save the model (e.g. 5 10 20)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    # Перевіряємо директорії
    train_dir = os.path.join(args.data_dir, "train")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    val_dir = os.path.join(args.data_dir, "val")
    do_val = os.path.isdir(val_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    # Трансформації
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # Датасети і лоадери
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    num_classes = len(train_dataset.classes)
    print(f"Detected {num_classes} classes: {train_dataset.classes}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=8, pin_memory=True)
    if do_val:
        val_dataset = datasets.ImageFolder(val_dir, transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=8, pin_memory=True)
    else:
        val_loader = None

    # Модель та пристрій
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Додамо змішані точності (AMP) для пришвидшення на GPU
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        # Додаємо тімінг однієї епохи
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            if use_amp:
                # Використовуємо автоматичне змішане точність (AMP)
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{args.epochs}, Train Loss: {avg_loss:.4f}")

        # Валідація (якщо є)
        if val_loader:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = model(images)
                    else:
                        outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            acc = correct / total * 100
            print(f"Validation Accuracy: {acc:.1f}%")

            # Зберігаємо модель з найкращою точністю
            if acc > best_acc:
                best_acc = acc
                # збережемо як best
                best_path = os.path.join(args.output_dir, f"resnet50_best.pth")
                torch.save(model.state_dict(), best_path)
                print(f"New best model saved to {best_path} ({acc:.1f}%)")

        # Якщо поточна епоха міститься в save_epochs, зберігаємо
        if epoch in args.save_epochs:
            model_filename = f"resnet50_{epoch}epochs.pth"
            model_path = os.path.join(args.output_dir, model_filename)
            torch.save(model.state_dict(), model_path)
            print(f"Model checkpoint saved to {model_path}")

    # Після завершення усіх епох: збережемо остаточну модель (якщо вона не збігається з уже збереженою)
    final_model_path = os.path.join(args.output_dir, f"resnet50_{args.epochs}epochs.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    # Збереження класів у JSON (якщо ще не існує або оновити)
    class_file = os.path.join(args.output_dir, "class_names.json")
    with open(class_file, 'w') as f:
        json.dump(train_dataset.classes, f)
    print(f"Class names saved to {class_file}")

if __name__ == "__main__":
    main()
