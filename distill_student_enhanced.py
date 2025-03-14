# distill_student_enhanced.py (continued)

    def eval_epoch(self, epoch):
        """Evaluate student model on validation set."""
        self.student.eval()
        
        total_loss = 0.0
        all_labels = []
        all_preds = []
        
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for imu_tensor, imu_mask, skel_tensor, skel_mask, labels in self.val_loader:
                # Move data to device
                imu_tensor = imu_tensor.to(self.device)
                imu_mask = imu_mask.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass for student
                student_outputs = self.student(imu_tensor, imu_mask)
                logits = student_outputs["logits"]
                
                # Calculate loss
                loss = criterion(logits, labels)
                total_loss += loss.item() * labels.size(0)
                
                # Get predictions
                _, predicted = torch.max(logits, dim=1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        val_loss = total_loss / len(all_labels) if all_labels else 0
        val_acc = 100.0 * np.mean(np.array(all_preds) == np.array(all_labels))
        
        # Calculate F1, precision, recall
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        if len(np.unique(all_labels)) <= 2:
            # Binary classification
            val_f1 = f1_score(all_labels, all_preds, average="binary")
            val_precision = precision_score(all_labels, all_preds, average="binary")
            val_recall = recall_score(all_labels, all_preds, average="binary")
        else:
            # Multi-class classification
            val_f1 = f1_score(all_labels, all_preds, average="macro")
            val_precision = precision_score(all_labels, all_preds, average="macro")
            val_recall = recall_score(all_labels, all_preds, average="macro")
        
        # Update metrics history
        self.metrics_history["val_loss"].append(val_loss)
        self.metrics_history["val_acc"].append(val_acc)
        self.metrics_history["val_f1"].append(val_f1)
        self.metrics_history["val_precision"].append(val_precision)
        self.metrics_history["val_recall"].append(val_recall)
        
        # Log metrics
        self.logger.info(
            f"Epoch {epoch+1} Val: "
            f"Loss={val_loss:.4f}, Acc={val_acc:.2f}%, "
            f"F1={val_f1:.4f}, Precision={val_precision:.4f}, Recall={val_recall:.4f}"
        )
        
        return val_loss, val_acc, val_f1, val_precision, val_recall
    def distill(self):
        """Start the enhanced distillation process."""
        # Get training parameters
        num_epochs = self.cfg.get("num_epoch", 50)
        
        # Initialize best validation metrics
        best_val_acc = 0.0
        best_val_f1 = 0.0
        best_epoch = 0
        
        # Initialize early stopping
        patience = self.cfg.get("early_stop_patience", 15)
        early_stop_counter = 0
        
        self.logger.info(f"Starting enhanced distillation for {num_epochs} epochs")
        
        # Training loop
        for epoch in range(num_epochs):
            # Train
            train_losses, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, val_f1, val_precision, val_recall = self.eval_epoch(epoch)
            
            # Update scheduler
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            elif self.scheduler is not None:
                self.scheduler.step()
            
            # Check if this is the best model
            is_best = False
            if val_f1 > best_val_f1:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                best_epoch = epoch + 1
                is_best = True
                
                # Save best model
                model_path = os.path.join(self.work_dir, f"{self.cfg.get('model_saved_name', 'student_enhanced_best')}.pth")
                torch.save(self.student.state_dict(), model_path)
                self.logger.info(f"New best model saved with F1: {best_val_f1:.4f}")
                
                # Reset early stopping counter
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            
            # Check for early stopping
            if early_stop_counter >= patience:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # Log epoch summary
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs} Summary: "
                f"Train Loss={train_losses['total']:.4f}, Train Acc={train_acc:.2f}%, "
                f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%, "
                f"F1={val_f1:.4f} "
                f"{'(Best)' if is_best else ''}"
            )
            
        # Save final model
        final_model_path = os.path.join(self.work_dir, f"{self.cfg.get('model_saved_name', 'student_enhanced')}_final.pth")
        torch.save(self.student.state_dict(), final_model_path)
        
        # Save training history
        history_path = os.path.join(self.work_dir, "distillation_history.json")
        import json
        with open(history_path, 'w') as f:
            # Convert numpy values to Python native types for JSON serialization
            serializable_history = {}
            for k, v in self.metrics_history.items():
                serializable_history[k] = [float(x) for x in v]
            
            json.dump(serializable_history, f, indent=2)
        
        # Plot training curves if matplotlib is available
        try:
            self.plot_training_curves()
        except ImportError:
            self.logger.info("Matplotlib not available, skipping training curves plotting")
        
        self.logger.info(
            f"Distillation completed. Best F1: {best_val_f1:.4f} at epoch {best_epoch}. "
            f"Final metrics - Accuracy: {best_val_acc:.2f}%, F1: {best_val_f1:.4f}"
        )
        
        return best_val_acc, best_val_f1
    
    def plot_training_curves(self):
        """Plot training curves for loss and metrics."""
        import matplotlib.pyplot as plt
        
        # Create figure for loss components
        plt.figure(figsize=(12, 8))
        
        # Plot loss components
        plt.subplot(2, 2, 1)
        plt.plot(self.metrics_history["train_loss_total"], label="Total Loss")
        plt.plot(self.metrics_history["train_loss_kl"], label="KL Loss")
        plt.plot(self.metrics_history["train_loss_ce"], label="CE Loss")
        plt.title("Training Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        # Plot feature and attention losses
        plt.subplot(2, 2, 2)
        plt.plot(self.metrics_history["train_loss_feat"], label="Feature Loss")
        plt.plot(self.metrics_history["train_loss_attn"], label="Attention Loss")
        plt.title("Feature and Attention Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        # Plot accuracy
        plt.subplot(2, 2, 3)
        plt.plot(self.metrics_history["train_acc"], label="Train Accuracy")
        plt.plot(self.metrics_history["val_acc"], label="Validation Accuracy")
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.grid(True)
        
        # Plot F1, precision, recall
        plt.subplot(2, 2, 4)
        plt.plot(self.metrics_history["val_f1"], label="F1 Score")
        plt.plot(self.metrics_history["val_precision"], label="Precision")
        plt.plot(self.metrics_history["val_recall"], label="Recall")
        plt.title("F1, Precision, Recall")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.work_dir, "distillation_curves.png"), dpi=150)
        plt.close()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Initialize the trainer
    trainer = EnhancedDistillTrainer(args)
    
    # Build data
    trainer.build_data()
    
    # Build models
    trainer.build_models()
    
    # Load teacher weights
    if not trainer.load_teacher_weights():
        print("Failed to load teacher weights. Aborting.")
        return
    
    # Freeze teacher model
    for param in trainer.teacher.parameters():
        param.requires_grad = False
    
    trainer.teacher.eval()
    
    # Build optimizer and loss
    trainer.build_optimizer_loss()
    
    # Start distillation
    trainer.distill()

if __name__ == "__main__":
    main() 
