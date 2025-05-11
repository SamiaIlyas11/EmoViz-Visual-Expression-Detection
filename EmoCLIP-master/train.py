import sys
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import confusion_matrix
import numpy as np
import warnings  # Import warnings module

# Suppress all PyTorch FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

from config import get_config
from DataLoaders import *
from architecture import VClip

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = True  # Enable for consistent input sizes

# Verify GPU availability
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Ensure your system has a compatible GPU.")

# Configuration
cnf = get_config(sys.argv)
cnf.dataset_name="Train_AFEW"
cnf.dataset_root = "/home/fast/Downloads/EmotiW_2018"
cnf.num_classes = 7
cnf.log_dir = "/home/fast/Downloads/EmotiW_2018/saved_models"
# Removed early stopping parameters
cnf.gradient_clip_val = 1.0  # New: gradient clipping value
cnf.mixup_alpha = 0.2  # New: mixup alpha parameter
os.makedirs(cnf.log_dir, exist_ok=True)

# Mixup function for data augmentation
def mixup_data(x, y, alpha=1.0):
    """Applies mixup augmentation to the batch"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Applies mixup to the loss calculation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train(loader, model, loss_criterion, optimizer, scaler, epoch, cnf):
    """
    Train the model on the training dataset with improved techniques.
    """
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []

    for batch_idx, data in enumerate(loader):
        # Handle batch data
        if len(data) == 3:
            inputs, labels, descriptions = data
        elif len(data) == 2:
            inputs, labels = data
            descriptions = None
        else:
            raise ValueError(f"Unexpected number of items in batch: {len(data)}")

        # Move inputs and labels to the correct device
        inputs = inputs.to(cnf.device)
        labels = labels.to(cnf.device)

        # Apply mixup if enabled
        if hasattr(cnf, 'use_mixup') and cnf.use_mixup and epoch > 5:  # Apply after a few epochs
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, cnf.mixup_alpha)
            inputs.requires_grad = True  # Ensure gradients are computed

        # Forward pass with mixed precision
        optimizer.zero_grad()
        
        # Suppress autocast deprecation warning
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            with autocast():
                logits = model(inputs, mode='classification')
                
                # Apply mixup loss if enabled
                if hasattr(cnf, 'use_mixup') and cnf.use_mixup and epoch > 5:
                    loss = mixup_criterion(loss_criterion, logits, labels_a, labels_b, lam)
                    # For tracking metrics only (not affecting loss)
                    predictions = logits.argmax(dim=1)
                    all_predictions.append(predictions.detach().cpu())
                    all_labels.append(labels.cpu())
                else:
                    loss = loss_criterion(logits, labels)
                    # Track metrics
                    predictions = logits.argmax(dim=1)
                    all_predictions.append(predictions.detach().cpu())
                    all_labels.append(labels.cpu())

        # Backpropagation with gradient scaling for mixed precision
        scaler.scale(loss).backward()
        
        # Gradient clipping to prevent exploding gradients
        if hasattr(cnf, 'gradient_clip_val') and cnf.gradient_clip_val > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cnf.gradient_clip_val)
        
        scaler.step(optimizer)
        scaler.update()

        # Update metrics
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:  # Print less frequently for speed
            print(f"Epoch {epoch+1}, Batch {batch_idx + 1}/{len(loader)}: Loss = {loss.item():.4f}")

    # Calculate epoch metrics
    all_predictions = torch.cat(all_predictions).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    # Compute confusion matrix for training data
    conf_matrix = confusion_matrix(all_labels, all_predictions, labels=range(cnf.num_classes))
    class_recalls = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    train_uar = np.mean(class_recalls)
    train_war = np.sum(conf_matrix.diagonal()) / np.sum(conf_matrix)
    
    return total_loss / len(loader), train_uar, train_war

@torch.no_grad()
def evaluate(loader, model, cnf):
    """
    Evaluate the model and compute metrics
    """
    model.eval()
    all_predictions = []
    all_labels = []
    val_loss = 0
    loss_criterion = nn.CrossEntropyLoss()

    for batch_idx, data in enumerate(loader):
        if len(data) == 3:
            inputs, labels, _ = data
        elif len(data) == 2:
            inputs, labels = data
        else:
            raise ValueError(f"Unexpected number of items in batch: {len(data)}")

        inputs = inputs.to(cnf.device)
        labels = labels.to(cnf.device)

        # Forward pass
        logits = model(inputs, mode='classification')
        loss = loss_criterion(logits, labels)
        val_loss += loss.item()
        
        predictions = logits.argmax(dim=1)

        all_predictions.append(predictions.cpu())
        all_labels.append(labels.cpu())

    # Concatenate predictions and labels
    all_predictions = torch.cat(all_predictions).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Compute confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions, labels=range(cnf.num_classes))
    
    # Compute metrics
    class_recalls = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    uar = np.mean(class_recalls)
    war = np.sum(conf_matrix.diagonal()) / np.sum(conf_matrix)

    print(f"Confusion Matrix:\n{conf_matrix}")
    
    print(f"UAR: {uar * 100:.2f}%, WAR: {war * 100:.2f}%")
    return uar, war, val_loss / len(loader)

if __name__ == "__main__":
    # Check if we're using distributed training
    use_distributed = "LOCAL_RANK" in os.environ and torch.cuda.device_count() > 1
    
    if use_distributed:
        # DDP setup
        cnf.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(cnf.local_rank)
        cnf.device = torch.device(f"cuda:{cnf.local_rank}")
        dist.init_process_group(backend="nccl", init_method="env://")
        cnf.world_size = dist.get_world_size()
        cnf.is_master = cnf.local_rank == 0
        # Set DDP flag for data loader
        cnf.DDP = True
    else:
        # Single GPU setup
        cnf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cnf.is_master = True
        cnf.world_size = 1
        cnf.local_rank = 0
        # Set DDP flag for data loader
        cnf.DDP = False
        print("Running in single GPU mode")
    
    # Set dataset configuration parameters
    cnf.use_augmentation = True  # Enable data augmentation
    cnf.use_mixup = True  # Enable mixup augmentation
    cnf.batch_size = getattr(cnf, 'batch_size', 32)  # Default batch size if not set
    cnf.clip_len = getattr(cnf, 'clip_len', 16)  # Default clip length if not set
    cnf.downsample = getattr(cnf, 'downsample', 1)  # Default downsample rate if not set
    
    # Load datasets
    train_loader, test_loader = get_loaders(cnf)
    
    # Initialize model
    model = VClip(num_classes=cnf.num_classes, num_layers=2).to(cnf.device)
    
    # Implement selective parameter freezing and learning rates
    if hasattr(cnf, 'finetune') and cnf.finetune:
        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False
        
        # Selectively unfreeze parameters based on configuration
        if hasattr(cnf, 'text') and cnf.text:
            for name, param in model.backbone.transformer.named_parameters():
                param.requires_grad = True
        
        if hasattr(cnf, 'visual') and cnf.visual:
            for name, param in model.backbone.visual.named_parameters():
                param.requires_grad = True
        
        # Separate parameter groups with weight decay exemptions
        no_decay = ['bias', 'LayerNorm.weight']
        backbone_params = []
        backbone_params_no_decay = []
        other_params = []
        other_params_no_decay = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
                
            if 'backbone' in name:
                if any(nd in name for nd in no_decay):
                    backbone_params_no_decay.append(param)
                else:
                    backbone_params.append(param)
            else:
                if any(nd in name for nd in no_decay):
                    other_params_no_decay.append(param)
                else:
                    other_params.append(param)
        
        # Create parameter groups with different learning rates
        param_groups = [
            {'params': other_params, 'lr': cnf.lr, 'weight_decay': cnf.wd},
            {'params': other_params_no_decay, 'lr': cnf.lr, 'weight_decay': 0.0},
            {'params': backbone_params, 'lr': cnf.lr * 0.01, 'weight_decay': cnf.wd},
            {'params': backbone_params_no_decay, 'lr': cnf.lr * 0.01, 'weight_decay': 0.0}
        ]
        
        # Initialize optimizer with parameter groups
        optimizer = torch.optim.AdamW(
            param_groups, 
            lr=cnf.lr, 
            weight_decay=cnf.wd
        )
    else:
        # If not finetuning, use AdamW optimizer with all parameters
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=cnf.lr, 
            weight_decay=cnf.wd
        )
    
    # Learning rate scheduler - Cosine Annealing with Warm Restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,  # Restart every 10 epochs
        T_mult=1, 
        eta_min=cnf.lr * 0.01
    )
    
    # Distributed Data Parallel setup if using distributed training
    if use_distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(
            model, 
            device_ids=[cnf.local_rank], 
            output_device=cnf.local_rank,
            find_unused_parameters=True
        )
    
    # Initialize loss criterion with label smoothing if configured
    if hasattr(cnf, 'use_label_smoothing') and cnf.use_label_smoothing:
        loss_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    else:
        loss_criterion = nn.CrossEntropyLoss()
    
    # Initialize the GradScaler - suppress the deprecation warning
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        scaler = GradScaler()
    
    # Initialize variables for tracking the best metrics
    best_war = 0.0
    best_uar = 0.0
    
    # Training loop without early stopping
    for epoch in range(cnf.num_epochs):
        print(f"Epoch {epoch + 1}/{cnf.num_epochs}:")
        
        # Set epoch for distributed samplers if applicable
        if use_distributed:
            if hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
            if hasattr(test_loader.sampler, 'set_epoch'):
                test_loader.sampler.set_epoch(epoch)
        
        # Print current learning rate
        if cnf.is_master:
            for param_group in optimizer.param_groups:
                print(f"Learning rate: {param_group['lr']:.6f}")
        
        # Training phase
        train_loss, train_uar, train_war = train(train_loader, model, loss_criterion, optimizer, scaler, epoch, cnf)
        print(f"Training Loss: {train_loss:.4f}, UAR: {train_uar*100:.2f}%, WAR: {train_war*100:.2f}%")
        
        # Update learning rate
        scheduler.step()
        
        # Validation phase
        if cnf.is_master:
            uar, war, val_loss = evaluate(test_loader, model, cnf)
            print(f"Validation Loss: {val_loss:.4f}, UAR: {uar * 100:.2f}%, WAR: {war * 100:.2f}%")
            
            # Check if this is the best model based on WAR
            if war > best_war:
                best_war = war
                best_model_path = os.path.join(cnf.log_dir, f"best_war_model_epoch_{epoch + 1}.pth")
                if use_distributed:
                    model_state_dict = model.module.state_dict()
                else:
                    model_state_dict = model.state_dict()
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_war': best_war,
                    'best_uar': uar,
                }, best_model_path)
                print(f"New best WAR model saved: '{best_model_path}' with WAR = {best_war * 100:.2f}%")
            
            # Also save based on UAR if it's the best
            if uar > best_uar:
                best_uar = uar
                best_uar_model_path = os.path.join(cnf.log_dir, f"best_uar_model_epoch_{epoch + 1}.pth")
                if use_distributed:
                    model_state_dict = model.module.state_dict()
                else:
                    model_state_dict = model.state_dict()
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_war': war,
                    'best_uar': best_uar,
                }, best_uar_model_path)
                print(f"New best UAR model saved: '{best_uar_model_path}' with UAR = {best_uar * 100:.2f}%")
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(cnf.log_dir, f"checkpoint_epoch_{epoch + 1}.pth")
                if use_distributed:
                    model_state_dict = model.module.state_dict()
                else:
                    model_state_dict = model.state_dict()
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_war': best_war,
                    'best_uar': best_uar,
                }, checkpoint_path)
                print(f"Checkpoint saved: '{checkpoint_path}'")
    
    # Final model save
    if cnf.is_master:
        final_model_path = os.path.join(cnf.log_dir, "final_model.pth")
        if use_distributed:
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_war': best_war,
            'best_uar': best_uar,
        }, final_model_path)
        print(f"Final model saved: '{final_model_path}' with Best WAR = {best_war * 100:.2f}%, Best UAR = {best_uar * 100:.2f}%")
    
    # Cleanup distributed process group if used
    if use_distributed:
        dist.destroy_process_group()
