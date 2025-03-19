if hasattr(self.model, 'forward_fusion') and fusion_features is not None:
                    logits = self.model.forward_fusion(acc_data.float(), fusion_features.float())
                elif hasattr(self.model, 'forward_quaternion') and quaternion is not None:
                    logits = self.model.forward_quaternion(acc_data.float(), quaternion.float())
                elif gyro_data is not None and hasattr(self.model, 'forward_multi_sensor'):
                    logits = self.model.forward_multi_sensor(acc_data.float(), gyro_data.float())
                else:
                    logits = self.model(acc_data.float())
                batch_loss = self.criterion(logits, targets)
                loss += batch_loss.sum().item()
                predictions = torch.argmax(F.log_softmax(logits, dim=1), 1)
                accuracy += (predictions == targets).sum().item()
                label_list.extend(targets.cpu().tolist())
                pred_list.extend(predictions.cpu().tolist())
                cnt += len(targets)
                process.set_postfix({
                    'loss': f"{loss/cnt:.4f}",
                    'acc': f"{100.0*accuracy/cnt:.2f}%"
                })
            loss /= cnt
            target = np.array(label_list)
            y_pred = np.array(pred_list)
            f1 = f1_score(target, y_pred, average='macro') * 100
            precision, recall, _, _ = precision_recall_fscore_support(target, y_pred, average='macro')
            precision *= 100
            recall *= 100
            balanced_acc = balanced_accuracy_score(target, y_pred) * 100
            accuracy *= 100. / cnt
            if result_file is not None:
                predict = pred_list
                true = label_list
                for i, x in enumerate(predict):
                    f_r.write(f"{x} ==> {true[i]}\n")
                f_r.close()
        self.print_log(
            f'{loader_name.capitalize()} metrics: Loss={loss:.4f}, '
            f'Accuracy={accuracy:.2f}%, F1={f1:.2f}, '
            f'Precision={precision:.2f}%, Recall={recall:.2f}%, '
            f'Balanced Accuracy={balanced_acc:.2f}%'
        )
        metrics = {
            'loss': loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'balanced_accuracy': balanced_acc,
            'false_alarm_rate': 100 - precision,
            'miss_rate': 100 - recall
        }
        if loader_name == 'val':
            save_model = False
            if f1 > self.best_f1:
                self.best_f1 = f1
                save_model = True
                self.print_log(f'New best model saved: improved validation F1 to {f1:.2f}')
            elif loss < self.best_loss:
                self.best_loss = loss
                save_model = True
                self.print_log(f'New best model saved: improved validation loss to {loss:.4f}')
            
            if save_model:
                self.best_accuracy = accuracy
                if isinstance(self.model, nn.DataParallel):
                    torch.save(deepcopy(self.model.module.state_dict()), self.model_path)
                else:
                    torch.save(deepcopy(self.model.state_dict()), self.model_path)
                if len(np.unique(target)) > 1:
                    self.cm_viz(y_pred, target)
        else:
            self.test_accuracy = accuracy
            self.test_f1 = f1
            self.test_precision = precision
            self.test_recall = recall
            self.test_balanced_accuracy = balanced_acc
            self.test_true = label_list
            self.test_pred = pred_list
        return loss, metrics

    def start(self):
        try:
            if self.arg.phase == 'train':
                self.train_loss_summary = []
                self.val_loss_summary = []
                self.train_metrics = []
                self.val_metrics = []
                self.best_accuracy = 0
                self.best_f1 = 0
                self.best_loss = float('inf')
                self.print_log(f'Parameters:\n{str(vars(self.arg))}\n')
                self.print_log(f'Starting training with {self.arg.optimizer} optimizer, LR={self.arg.base_lr}')
                if hasattr(self.arg, 'dataset_args') and isinstance(self.arg.dataset_args, dict) and 'fusion_options' in self.arg.dataset_args:
                    filter_type = self.arg.dataset_args['fusion_options'].get('filter_type', 'unknown')
                    self.print_log(f"Using filter type: {filter_type}")
                results = self.create_df(columns=['fold', 'test_subject', 'train_subjects', 'accuracy', 'f1_score', 'precision', 'recall'])
                use_kfold = False
                fold_assignments = []
                if hasattr(self.arg, 'kfold') and self.arg.kfold:
                    use_kfold = True
                    if hasattr(self.arg, 'kfold_assignments'):
                        fold_assignments = self.arg.kfold_assignments
                        self.print_log(f"Using provided fold assignments with {len(fold_assignments)} folds")
                elif hasattr(self.arg, 'kfold') and isinstance(self.arg.kfold, dict):
                    use_kfold = self.arg.kfold.get('enabled', False)
                    if use_kfold and 'fold_assignments' in self.arg.kfold:
                        fold_assignments = self.arg.kfold.get('fold_assignments', [])
                        self.print_log(f"Using provided fold assignments with {len(fold_assignments)} folds")
                if use_kfold and not fold_assignments:
                    all_subjects = self.arg.subjects.copy()
                    num_folds = getattr(self.arg, 'num_folds', 5)
                    if hasattr(self.arg, 'kfold') and isinstance(self.arg.kfold, dict):
                        num_folds = self.arg.kfold.get('num_folds', 5)
                    np.random.seed(self.arg.seed)
                    np.random.shuffle(all_subjects)
                    fold_size = len(all_subjects) // num_folds
                    for i in range(num_folds):
                        start_idx = i * fold_size
                        end_idx = start_idx + fold_size if i < num_folds - 1 else len(all_subjects)
                        fold_assignments.append(all_subjects[start_idx:end_idx])
                    self.print_log(f"Created {num_folds} automatic fold assignments")
                if use_kfold:
                    self.print_log(f"Starting {len(fold_assignments)}-fold cross-validation")
                    fold_metrics = []
                    all_subjects = self.arg.subjects.copy()
                    for fold_idx, test_subjects in enumerate(fold_assignments):
                        self.print_log(f"\n{'='*20} Starting Fold {fold_idx+1}/{len(fold_assignments)} {'='*20}")
                        self.best_loss = float('inf')
                        self.best_accuracy = 0
                        self.best_f1 = 0
                        self.patience_counter = 0
                        next_fold_idx = (fold_idx + 1) % len(fold_assignments)
                        self.val_subject = fold_assignments[next_fold_idx]
                        self.test_subject = test_subjects
                        self.train_subjects = []
                        for i, fold in enumerate(fold_assignments):
                            if i != fold_idx and i != next_fold_idx:
                                self.train_subjects.extend(fold)
                        self.print_log(f'Fold {fold_idx+1}: Test subjects={self.test_subject}')
                        self.print_log(f'Validation subjects={self.val_subject}')
                        self.print_log(f'Training subjects={self.train_subjects}')
                        self.model = self.load_model(self.arg.model, self.arg.model_args)
                        if len(self.available_gpus) > 1 and self.arg.multi_gpu:
                            self.model = nn.DataParallel(
                                self.model, 
                                device_ids=self.available_gpus
                            )
                        self.print_log(f"Loading data for fold {fold_idx+1}...")
                        if not self.load_data():
                            self.print_log(f"ERROR: Failed to load data for fold {fold_idx+1}")
                            continue
                        train_batches = len(self.data_loader.get('train', [])) if 'train' in self.data_loader else 0
                        val_batches = len(self.data_loader.get('val', [])) if 'val' in self.data_loader else 0
                        test_batches = len(self.data_loader.get('test', [])) if 'test' in self.data_loader else 0
                        self.print_log(f"Data loaded: {train_batches} training batches, "
                                    f"{val_batches} validation batches, "
                                    f"{test_batches} test batches")
                        self.load_optimizer()
                        self.print_log(f"Starting training for fold {fold_idx+1}...")
                        self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
                        patience = self.arg.patience
                        for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                            val_loss, val_f1 = self.train(epoch)
                            if self.early_stopping(val_loss, val_f1, epoch):
                                self.print_log(f"Early stopping triggered after {epoch+1} epochs")
                                break
                        if len(self.train_loss_summary) > 0 and len(self.val_loss_summary) > 0:
                            self.loss_viz(self.train_loss_summary, self.val_loss_summary)
                            self.print_log(f"Loss curves saved to {self.arg.work_dir}/train_vs_val_loss.png")
                        self.print_log(f'Training complete for fold {fold_idx+1}, loading best model for testing')
                        if os.path.exists(self.model_path):
                            test_model = self.load_model(self.arg.model, self.arg.model_args)
                            if len(self.available_gpus) > 1 and self.arg.multi_gpu:
                                test_model = nn.DataParallel(
                                    test_model, 
                                    device_ids=self.available_gpus
                                )
                            try:
                                if isinstance(test_model, nn.DataParallel):
                                    test_model.module.load_state_dict(torch.load(self.model_path))
                                else:
                                    test_model.load_state_dict(torch.load(self.model_path))
                                self.print_log(f"Successfully loaded best model from {self.model_path}")
                                self.model = test_model
                            except Exception as e:
                                self.print_log(f"WARNING: Could not load best model: {str(e)}")
                                self.print_log("Using current model state for testing")
                        else:
                            self.print_log(f"WARNING: No saved model found at {self.model_path}")
                            self.print_log("Using current model state for testing")
                        self.model.eval()
                        if 'test' in self.data_loader:
                            self.print_log(f'------ Testing on subjects {self.test_subject} ------')
                            test_loss, test_metrics = self.eval(epoch=0, loader_name='test')
                            fold_result = {
                                'fold': fold_idx + 1,
                                'test_subjects': self.test_subject,
                                'accuracy': self.test_accuracy,
                                'f1': self.test_f1,
                                'precision': self.test_precision,
                                'recall': self.test_recall,
                                'balanced_accuracy': self.test_balanced_accuracy
                            }
                            fold_metrics.append(fold_result)
                            subject_result = pd.Series({
                                'fold': fold_idx + 1,
                                'test_subject': str(self.test_subject),
                                'train_subjects': str(self.train_subjects),
                                'accuracy': round(self.test_accuracy, 2),
                                'f1_score': round(self.test_f1, 2),
                                'precision': round(self.test_precision, 2),
                                'recall': round(self.test_recall, 2)
                            })
                            results.loc[len(results)] = subject_result
                            fold_dir = os.path.join(self.arg.work_dir, f"fold_{fold_idx+1}")
                            os.makedirs(fold_dir, exist_ok=True)
                            if hasattr(self, 'cm_viz') and hasattr(self, 'test_pred') and hasattr(self, 'test_true'):
                                try:
                                    self.cm_viz(np.array(self.test_pred), np.array(self.test_true))
                                    fold_cm_path = os.path.join(fold_dir, "confusion_matrix.png")
                                    shutil.copy(os.path.join(self.arg.work_dir, "confusion_matrix.png"), fold_cm_path)
                                    self.print_log(f"Saved fold-specific confusion matrix to {fold_cm_path}")
                                except Exception as e:
                                    self.print_log(f"Error saving confusion matrix: {str(e)}")
                        else:
                            self.print_log(f"WARNING: No test data loader available for fold {fold_idx+1}")
                        self.train_loss_summary = []
                        self.val_loss_summary = []
                    if fold_metrics:
                        avg_metrics = {
                            'accuracy': np.mean([m['accuracy'] for m in fold_metrics]),
                            'accuracy_std': np.std([m['accuracy'] for m in fold_metrics]),
                            'f1': np.mean([m['f1'] for m in fold_metrics]),
                            'f1_std': np.std([m['f1'] for m in fold_metrics]),
                            'precision': np.mean([m['precision'] for m in fold_metrics]),
                            'precision_std': np.std([m['precision'] for m in fold_metrics]),
                            'recall': np.mean([m['recall'] for m in fold_metrics]),
                            'recall_std': np.std([m['recall'] for m in fold_metrics]),
                            'balanced_accuracy': np.mean([m['balanced_accuracy'] for m in fold_metrics]),
                            'balanced_accuracy_std': np.std([m['balanced_accuracy'] for m in fold_metrics])
                        }
                        filter_type = "unknown"
                        if hasattr(self.arg, 'dataset_args') and isinstance(self.arg.dataset_args, dict) and 'fusion_options' in self.arg.dataset_args:
                            filter_type = self.arg.dataset_args['fusion_options'].get('filter_type', 'unknown')
                        cv_summary = {
                            'fold_metrics': fold_metrics,
                            'average_metrics': avg_metrics,
                            'filter_type': filter_type
                        }
                        with open(os.path.join(self.arg.work_dir, 'cv_summary.json'), 'w') as f:
                            json.dump(cv_summary, f, indent=2)
                        self.print_log(f"Cross-validation summary saved to {self.arg.work_dir}/cv_summary.json")
                        self.print_log(f'\n===== Cross-Validation Results =====')
                        self.print_log(f'Mean accuracy: {avg_metrics["accuracy"]:.2f}% ± {avg_metrics["accuracy_std"]:.2f}%')
                        self.print_log(f'Mean F1 score: {avg_metrics["f1"]:.2f} ± {avg_metrics["f1_std"]:.2f}')
                        self.print_log(f'Mean precision: {avg_metrics["precision"]:.2f}% ± {avg_metrics["precision_std"]:.2f}%')
                        self.print_log(f'Mean recall: {avg_metrics["recall"]:.2f}% ± {avg_metrics["recall_std"]:.2f}%')
                        self.print_log(f'Mean balanced accuracy: {avg_metrics["balanced_accuracy"]:.2f}% ± {avg_metrics["balanced_accuracy_std"]:.2f}%')
                    results.to_csv(os.path.join(self.arg.work_dir, 'fold_scores.csv'), index=False)
                    self.print_log(f"Fold-specific scores saved to {self.arg.work_dir}/fold_scores.csv")
                else:
                    self.print_log("Starting standard train/val/test split training (no cross-validation)")
                    if not self.train_subjects and not self.val_subject and not self.test_subject:
                        total_subjects = len(self.arg.subjects)
                        test_idx = max(1, total_subjects // 5)
                        val_idx = test_idx * 2
                        self.test_subject = self.arg.subjects[0:test_idx]
                        self.val_subject = self.arg.subjects[test_idx:val_idx]
                        self.train_subjects = self.arg.subjects[val_idx:]
                    self.print_log(f'Test subjects: {self.test_subject}')
                    self.print_log(f'Validation subjects: {self.val_subject}')
                    self.print_log(f'Training subjects: {self.train_subjects}')
                    if not self.load_data():
                        self.print_log("WARNING: Data loading issues encountered - will attempt to continue with available data")
                    self.load_optimizer()
                    self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
                    patience = self.arg.patience
                    for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                        val_loss, val_f1 = self.train(epoch)
                        if self.early_stopping(val_loss, val_f1, epoch):
                            self.print_log(f"Early stopping triggered after {epoch+1} epochs")
                            break
                    if len(self.train_loss_summary) > 0 and len(self.val_loss_summary) > 0:
                        self.loss_viz(self.train_loss_summary, self.val_loss_summary)
                    self.print_log('Training complete, loading best model for testing')
                    if os.path.exists(self.model_path):
                        test_model = self.load_model(self.arg.model, self.arg.model_args)
                        if len(self.available_gpus) > 1 and self.arg.multi_gpu:
                            test_model = nn.DataParallel(
                                test_model, 
                                device_ids=self.available_gpus
                            )
                        try:
                            if isinstance(test_model, nn.DataParallel):
                                test_model.module.load_state_dict(torch.load(self.model_path))
                            else:
                                test_model.load_state_dict(torch.load(self.model_path))
                            self.model = test_model
                            self.print_log("Successfully loaded best model weights")
                        except Exception as e:
                            self.print_log(f"WARNING: Could not load best model: {str(e)}")
                            self.print_log("Using current model state for testing")
                    else:
                        self.print_log(f"WARNING: No saved model found at {self.model_path}")
                        self.print_log("Using current model state for testing")
                    self.model.eval()
                    if 'test' in self.data_loader:
                        self.print_log(f'------ Testing on subjects {self.test_subject} ------')
                        self.eval(epoch=0, loader_name='test')
                        test_result = pd.Series({
                            'test_subject': str(self.test_subject),
                            'train_subjects': str(self.train_subjects),
                            'accuracy': round(self.test_accuracy, 2),
                            'f1_score': round(self.test_f1, 2),
                            'precision': round(self.test_precision, 2),
                            'recall': round(self.test_recall, 2)
                        })
                        results.loc[len(results)] = test_result
                        results.to_csv(os.path.join(self.arg.work_dir, 'test_scores.csv'), index=False)
            else:
                self.print_log('Testing mode - evaluating pre-trained model')
                if not hasattr(self, 'test_subject') or not self.test_subject:
                    self.test_subject = self.arg.subjects
                if not self.load_data():
                    self.print_log("ERROR: Failed to load test data")
                    return
                self.eval(epoch=0, loader_name='test', result_file=self.arg.result_file)
                self.print_log(f'Test results: Accuracy={self.test_accuracy:.2f}%, F1={self.test_f1:.4f}, '
                            f'Precision={self.test_precision:.4f}, Recall={self.test_recall:.4f}')
                filter_type = "unknown"
                if hasattr(self.arg, 'dataset_args') and isinstance(self.arg.dataset_args, dict) and 'fusion_options' in self.arg.dataset_args:
                    filter_type = self.arg.dataset_args['fusion_options'].get('filter_type', 'unknown')
                test_results = {
                    'filter_type': filter_type,
                    'accuracy': self.test_accuracy,
                    'f1': self.test_f1,
                    'precision': self.test_precision,
                    'recall': self.test_recall,
                    'balanced_accuracy': self.test_balanced_accuracy
                }
                with open(os.path.join(self.arg.work_dir, 'test_results.json'), 'w') as f:
                    json.dump(test_results, f, indent=2)
                self.print_log(f"Test results saved to {self.arg.work_dir}/test_results.json")
                if hasattr(self, 'cm_viz') and hasattr(self, 'test_pred') and hasattr(self, 'test_true'):
                    try:
                        self.cm_viz(np.array(self.test_pred), np.array(self.test_true))
                        self.print_log(f"Confusion matrix saved to {self.arg.work_dir}/confusion_matrix.png")
                    except Exception as e:
                        self.print_log(f"Error creating confusion matrix: {str(e)}")
        except Exception as e:
            self.print_log(f"ERROR in training/testing workflow: {str(e)}")
            self.print_log(traceback.format_exc())
            if hasattr(self, 'model') and self.arg.phase == 'train':
                emergency_path = os.path.join(self.arg.work_dir, 'emergency_checkpoint.pt')
                try:
                    if isinstance(self.model, nn.DataParallel):
                        torch.save(self.model.module.state_dict(), emergency_path)
                    else:
                        torch.save(self.model.state_dict(), emergency_path)
                    self.print_log(f"Saved emergency checkpoint to {emergency_path}")
                except Exception as save_error:
                    self.print_log(f"Could not save emergency checkpoint: {str(save_error)}")

    def save_results(self):
        try:
            if not hasattr(self, 'train_losses'):
                self.train_losses = []
            if not hasattr(self, 'val_losses'):
                self.val_losses = []
            if not hasattr(self, 'train_metrics'):
                self.train_metrics = []
            if not hasattr(self, 'val_metrics'):
                self.val_metrics = []
            max_len = max(
                len(self.train_loss_summary) if hasattr(self, 'train_loss_summary') else 0,
                len(self.val_loss_summary) if hasattr(self, 'val_loss_summary') else 0,
                len(self.train_metrics) if hasattr(self, 'train_metrics') else 0,
                len(self.val_metrics) if hasattr(self, 'val_metrics') else 0
            )
            if max_len == 0:
                return
            epochs = list(range(max_len))
            train_loss = self.train_loss_summary + [None] * (max_len - len(self.train_loss_summary))
            val_loss = self.val_loss_summary + [None] * (max_len - len(self.val_loss_summary))
            train_acc = [m.get('accuracy', None) if m else None for m in self.train_metrics] + [None] * (max_len - len(self.train_metrics))
            train_f1 = [m.get('f1', None) if m else None for m in self.train_metrics] + [None] * (max_len - len(self.train_metrics))
            val_acc = [m.get('accuracy', None) if m else None for m in self.val_metrics] + [None] * (max_len - len(self.val_metrics))
            val_f1 = [m.get('f1', None) if m else None for m in self.val_metrics] + [None] * (max_len - len(self.val_metrics))
            metrics_df = pd.DataFrame({
                'epoch': epochs,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_accuracy': train_acc,
                'train_f1': train_f1,
                'val_accuracy': val_acc,
                'val_f1': val_f1
            })
            metrics_path = os.path.join(self.arg.work_dir, 'training_metrics.csv')
            metrics_df.to_csv(metrics_path, index=False)
            self.print_log(f"Saved training metrics to {metrics_path}")
            self.plot_metric_curves()
        except Exception as e:
            self.print_log(f"Error saving results: {str(e)}")
            self.print_log(traceback.format_exc())

    def plot_metric_curves(self):
        try:
            metrics_to_plot = [
                ('loss', 'Loss'),
                ('accuracy', 'Accuracy'),
                ('f1', 'F1 Score')
            ]
            for metric, title in metrics_to_plot:
                plt.figure(figsize=(10, 6))
                if metric == 'loss':
                    plt.plot(self.train_loss_summary, 'b-', label=f'Training {title}')
                    plt.plot(self.val_loss_summary, 'r-', label=f'Validation {title}')
                else:
                    train_values = [m.get(metric, None) if m else None for m in self.train_metrics]
                    val_values = [m.get(metric, None) if m else None for m in self.val_metrics]
                    if any(v is not None for v in train_values):
                        plt.plot(train_values, 'b-', label=f'Training {title}')
                    if any(v is not None for v in val_values):
                        plt.plot(val_values, 'r-', label=f'Validation {title}')
                plt.xlabel('Epoch')
                plt.ylabel(title)
                plt.title(f'Training and Validation {title}')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(self.arg.work_dir, f'{metric}_curve.png'))
                plt.close()
        except Exception as e:
            self.print_log(f"Error plotting metric curves: {str(e)}")

def main():
    parser = get_args()
    arg = parser.parse_args()
    if arg.config is not None:
        with open(arg.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(arg).keys()
        for k in default_arg.keys():
            if k not in key:
                print('Unknown Arguments: {}'.format(k))
                assert k in key
        parser.set_defaults(**default_arg)
        arg = parser.parse_args()
    init_seed(arg.seed)
    trainer = Trainer(arg)
    if arg.phase == 'train':
        trainer.start()
    elif arg.phase == 'test':
        if arg.weights is None:
            raise ValueError('Please appoint --weights.')
        trainer.test_subject = arg.subjects
        trainer.start()
    else:
        raise ValueError('Unknown phase: ' + arg.phase)

if __name__ == '__main__':
    main()
