def train(loader, model, loss_criterion, optimizer, scaler, epoch, cnf):
    """
    Train the model on the training dataset with improved techniques and EmoCLIP's class descriptions.
    """
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []

    # Load emotion descriptions from YAML
    class_descr = yaml.safe_load(Path('DataLoaders/class_descriptions.yml').read_text())
    class_descriptions = [class_descr[cls.lower()] for cls in CLASSES]
    
    for batch_idx, data in enumerate(loader):
        # Handle batch data
        if len(data) == 3:
            inputs, labels, _ = data
        elif len(data) == 2:
            inputs, labels = data
        else:
            raise ValueError(f"Unexpected number of items in batch: {len(data)}")
            
        # Move inputs and labels to the correct device
        inputs = inputs.to(cnf.device)
        labels = labels.to(cnf.device)
        
        # Get the rich descriptions for each sample based on its label
        batch_descriptions = [class_descriptions[label.item()] for label in labels]
        class_tokens = clip.tokenize(batch_descriptions, context_length=77, truncate=True).to(cnf.device)
        
        # Apply mixup if enabled
        if hasattr(cnf, 'use_mixup') and cnf.use_mixup and epoch > 5:
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, cnf.mixup_alpha)
            # Also mix the text tokens if using mixup
            indices_a = torch.arange(len(inputs), device=cnf.device)
            indices_b = torch.randperm(len(inputs), device=cnf.device)
            mixed_tokens = lam * class_tokens + (1 - lam) * class_tokens[indices_b]
            inputs.requires_grad = True
        else:
            mixed_tokens = class_tokens
            
        # Forward pass with mixed precision
        optimizer.zero_grad()
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            with autocast():
                # Use contrastive mode instead of classification
                logits_per_image, logits_per_text = model(inputs, mixed_tokens, mode='similarity')
                
                # Contrastive learning loss
                ground_truth = torch.arange(len(inputs), device=cnf.device)
                
                # If using mixup, adjust contrastive loss
                if hasattr(cnf, 'use_mixup') and cnf.use_mixup and epoch > 5:
                    # Mix image-to-text loss
                    loss_i = lam * loss_criterion(logits_per_image, ground_truth) + \
                             (1 - lam) * loss_criterion(logits_per_image, indices_b)
                    # Mix text-to-image loss
                    loss_t = lam * loss_criterion(logits_per_text, ground_truth) + \
                             (1 - lam) * loss_criterion(logits_per_text, indices_b)
                else:
                    loss_i = loss_criterion(logits_per_image, ground_truth)
                    loss_t = loss_criterion(logits_per_text, ground_truth)
                
                loss = (loss_i + loss_t) / 2
                
                # For tracking metrics
                predicted = logits_per_image.softmax(dim=-1).argmax(dim=-1)
                all_predictions.append(predicted.detach().cpu())
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
        
        if batch_idx % 10 == 0:
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
