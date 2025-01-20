# train.py

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.config import *
from src.data.data_loader import EmotionDataLoader
from src.models.efficient_face import EfficientFaceResNet
from src.training.utils import EarlyStopping, accuracy, AverageMeter
from src.visualization.visualize import plot_training_curves, plot_confusion_matrix

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()

    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, targets)

        acc1, = accuracy(outputs, targets, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg, top1.avg

def validate(model, val_loader, criterion, device):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)

            acc1, = accuracy(outputs, targets, topk=(1,))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))

    return losses.avg, top1.avg

def main():
    # Initialize data loaders
    data_loader = EmotionDataLoader(TRAIN_DIR, TEST_DIR, BATCH_SIZE)
    train_loader, test_loader = data_loader.get_loaders()

    # Initialize model, criterion, optimizer
    model = EfficientFaceResNet(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=LR_SCHEDULER_PATIENCE,
        factor=LR_SCHEDULER_FACTOR,
        min_lr=MIN_LR,
        verbose=True
    )

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, verbose=True)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0

    # Training loop
    for epoch in range(NUM_EPOCHS):
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(model, test_loader, criterion, DEVICE)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'New best model saved with validation accuracy: {best_val_acc:.2f}%')
        
        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Load best model and evaluate
    model.load_state_dict(torch.load('best_model.pth'))
    test_loss, test_acc = validate(model, test_loader, criterion, DEVICE)
    print(f'Final Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

    # Plot training curves and confusion matrix
    plot_training_curves(history)
    plot_confusion_matrix(model, test_loader, DEVICE, EMOTION_LABELS)

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': best_val_acc,
        'history': history
    }, 'emotion_recognition_model.pth')

if __name__ == '__main__':
    main()
