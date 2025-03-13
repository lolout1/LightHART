#!/usr/bin/env python
"""
Analyze cross-modal knowledge transfer from skeleton to IMU.
"""

import argparse
import yaml
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from importlib import import_module
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze cross-modal knowledge transfer."
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/smartfallmm/distill_student_enhanced.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--teacher-weights", 
        type=str, 
        default="exps/teacher_quat/teacher_quat_best.pth",
        help="Path to teacher model weights"
    )
    parser.add_argument(
        "--student-weights", 
        type=str, 
        default="exps/student_enhanced/student_enhanced_best.pth",
        help="Path to student model weights"
    )
    parser.add_argument(
        "--val-batch", 
        type=str, 
        default="",
        help="Path to validation batch data (if not provided, will use a random batch)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./cross_modal_analysis",
        help="Directory to save analysis results"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="0",
        help="GPU device ID"
    )
    return parser.parse_args()

class FeatureVisualizer:
    """Visualize features from teacher and student models."""
    
    def __init__(self, arg):
        """
        Initialize visualizer.
        
        Args:
            arg: Command line arguments
        """
        # Load configuration
        with open(arg.config, 'r') as f:
            self.cfg = yaml.safe_load(f)
        
        # Set device
        os.environ["CUDA_VISIBLE_DEVICES"] = arg.device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Create output directory
        self.output_dir = arg.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Model and weights paths
        self.teacher_weights = arg.teacher_weights
        self.student_weights = arg.student_weights
        
        # Validation batch path
        self.val_batch = arg.val_batch
    
    def load_models(self):
        """Load teacher and student models."""
        # Import model classes
        teacher_path = self.cfg["teacher_model"]
        student_path = self.cfg["student_model"]
        
        t_parts = teacher_path.split(".")
        t_module = ".".join(t_parts[:-1])
        t_class = t_parts[-1]
        
        s_parts = student_path.split(".")
        s_module = ".".join(s_parts[:-1])
        s_class = s_parts[-1]
        
        t_mod = import_module(t_module)
        TeacherClass = getattr(t_mod, t_class)
        
        s_mod = import_module(s_module)
        StudentClass = getattr(s_mod, s_class)
        
        # Create models
        self.teacher = TeacherClass(**self.cfg["teacher_args"]).to(self.device)
        self.student = StudentClass(**self.cfg["student_args"]).to(self.device)
        
        # Load weights
        if os.path.exists(self.teacher_weights):
            state_dict = torch.load(self.teacher_weights, map_location=self.device)
            self.teacher.load_state_dict(state_dict, strict=False)
            print(f"Loaded teacher weights from {self.teacher_weights}")
        else:
            print(f"Warning: Teacher weights not found at {self.teacher_weights}")
        
        if os.path.exists(self.student_weights):
            state_dict = torch.load(self.student_weights, map_location=self.device)
            self.student.load_state_dict(state_dict)
            print(f"Loaded student weights from {self.student_weights}")
        else:
            print(f"Warning: Student weights not found at {self.student_weights}")
        
        # Set models to eval mode
        self.teacher.eval()
        self.student.eval()
    
    def prepare_data(self):
        """Prepare data for visualization."""
        if self.val_batch and os.path.exists(self.val_batch):
            # Load validation batch
            data = torch.load(self.val_batch, map_location=self.device)
            return data
        else:
            # Create random data
            print("Warning: Validation batch not provided, using random data")
            
            batch_size = 16
            seq_len = 128
            
            # Create random IMU data
            imu_tensor = torch.randn(batch_size, seq_len, 16).to(self.device)
            imu_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool).to(self.device)
            
            # Create random skeleton data
            skel_tensor = torch.randn(batch_size, seq_len, 96).to(self.device)
            skel_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool).to(self.device)
            
            # Create random labels
            labels = torch.randint(0, 2, (batch_size,)).to(self.device)
            
            return {
                "imu_tensor": imu_tensor,
                "imu_mask": imu_mask,
                "skel_tensor": skel_tensor,
                "skel_mask": skel_mask,
                "labels": labels
            }
    
    def extract_features(self, data):
        """
        Extract features from teacher and student models.
        
        Args:
            data: Batch data
            
        Returns:
            Dictionary with extracted features
        """
        # Unpack data
        imu_tensor = data["imu_tensor"]
        imu_mask = data["imu_mask"]
        skel_tensor = data["skel_tensor"]
        skel_mask = data["skel_mask"]
        labels = data["labels"]
        
        # Get teacher features
        with torch.no_grad():
            teacher_outputs = self.teacher(skel_tensor, imu_tensor, skel_mask, imu_mask)
            student_outputs = self.student(imu_tensor, imu_mask)
        
        # Extract features
        features = {
            "teacher_skel_feat": teacher_outputs["skel_feat"].cpu().numpy(),
            "teacher_imu_feat": teacher_outputs["imu_feat"].cpu().numpy(),
            "teacher_fused_feat": teacher_outputs["fused_feat"].cpu().numpy(),
            "student_feat": student_outputs["feat"].cpu().numpy(),
            "labels": labels.cpu().numpy()
        }
        
        # Extract attention maps (if available)
        if "imu_attentions" in teacher_outputs and "attentions" in student_outputs:
            # For simplicity, take the last layer attention maps
            teacher_attn = teacher_outputs["imu_attentions"][-1]  # (batch, heads, seq, seq)
            student_attn = student_outputs["attentions"][-1]
            
            # Average across attention heads
            teacher_attn = teacher_attn.mean(dim=1).cpu().numpy()  # (batch, seq, seq)
            student_attn = student_attn.mean(dim=1).cpu().numpy()
            
            features["teacher_attn"] = teacher_attn
            features["student_attn"] = student_attn
        
        return features
    
    def visualize_feature_space(self, features):
        """
        Visualize feature spaces using dimensionality reduction.
        
        Args:
            features: Dictionary with extracted features
        """
        # Create scatter plot using PCA
        pca = PCA(n_components=2)
        
        plt.figure(figsize=(12, 10))
        
        # Combine all features for PCA fitting
        all_feats = np.vstack([
            features["teacher_skel_feat"],
            features["teacher_imu_feat"],
            features["teacher_fused_feat"],
            features["student_feat"]
        ])
        
        # Fit PCA
        pca.fit(all_feats)
        
        # Transform each feature set
        teacher_skel_pca = pca.transform(features["teacher_skel_feat"])
        teacher_imu_pca = pca.transform(features["teacher_imu_feat"])
        teacher_fused_pca = pca.transform(features["teacher_fused_feat"])
        student_pca = pca.transform(features["student_feat"])
        
        # Get labels
        labels = features["labels"]
        
        # Define colors for each label
        colors = ['blue', 'red']
        
        # Create scatter plot with PCA
        plt.subplot(2, 2, 1)
        for i in range(len(colors)):
            idx = labels == i
            if np.any(idx):
                plt.scatter(
                    teacher_skel_pca[idx, 0],
                    teacher_skel_pca[idx, 1],
                    c=colors[i],
                    label=f"Class {i}",
                    marker='o',
                    alpha=0.7
                )
        plt.title("Teacher Skeleton Features (PCA)")
        plt.legend()
        
        plt.subplot(2, 2, 2)
        for i in range(len(colors)):
            idx = labels == i
            if np.any(idx):
                plt.scatter(
                    teacher_imu_pca[idx, 0],
                    teacher_imu_pca[idx, 1],
                    c=colors[i],
                    label=f"Class {i}",
                    marker='^',
                    alpha=0.7
                )
        plt.title("Teacher IMU Features (PCA)")
        plt.legend()
        
        plt.subplot(2, 2, 3)
        for i in range(len(colors)):
            idx = labels == i
            if np.any(idx):
                plt.scatter(
                    teacher_fused_pca[idx, 0],
                    teacher_fused_pca[idx, 1],
                    c=colors[i],
                    label=f"Class {i}",
                    marker='s',
                    alpha=0.7
                )
        plt.title("Teacher Fused Features (PCA)")
        plt.legend()
        
        plt.subplot(2, 2, 4)
        for i in range(len(colors)):
            idx = labels == i
            if np.any(idx):
                plt.scatter(
                    student_pca[idx, 0],
                    student_pca[idx, 1],
                    c=colors[i],
                    label=f"Class {i}",
                    marker='*',
                    alpha=0.7
                )
        plt.title("Student Features (PCA)")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "feature_space_pca.png"), dpi=150)
        plt.close()
        
        # Create t-SNE visualization
        tsne = TSNE(n_components=2, random_state=42)
        
        plt.figure(figsize=(12, 10))
        
        # Transform each feature set with t-SNE (separately for better visualization)
        teacher_skel_tsne = tsne.fit_transform(features["teacher_skel_feat"])
        teacher_imu_tsne = tsne.fit_transform(features["teacher_imu_feat"])
        teacher_fused_tsne = tsne.fit_transform(features["teacher_fused_feat"])
        student_tsne = tsne.fit_transform(features["student_feat"])
        
        # Create scatter plot with t-SNE
        plt.subplot(2, 2, 1)
        for i in range(len(colors)):
            idx = labels == i
            if np.any(idx):
                plt.scatter(
                    teacher_skel_tsne[idx, 0],
                    teacher_skel_tsne[idx, 1],
                    c=colors[i],
                    label=f"Class {i}",
                    marker='o',
                    alpha=0.7
                )
        plt.title("Teacher Skeleton Features (t-SNE)")
        plt.legend()
        
        plt.subplot(2, 2, 2)
        for i in range(len(colors)):
            idx = labels == i
            if np.any(idx):
                plt.scatter(
                    teacher_imu_tsne[idx, 0],
                    teacher_imu_tsne[idx, 1],
                    c=colors[i],
                    label=f"Class {i}",
                    marker='^',
                    alpha=0.7
                )
        plt.title("Teacher IMU Features (t-SNE)")
        plt.legend()
        
        plt.subplot(2, 2, 3)
        for i in range(len(colors)):
            idx = labels == i
            if np.any(idx):
                plt.scatter(
                    teacher_fused_tsne[idx, 0],
                    teacher_fused_tsne[idx, 1],
                    c=colors[i],
                    label=f"Class {i}",
                    marker='s',
                    alpha=0.7
                )
        plt.title("Teacher Fused Features (t-SNE)")
        plt.legend()
        
        plt.subplot(2, 2, 4)
        for i in range(len(colors)):
            idx = labels == i
            if np.any(idx):
                plt.scatter(
                    student_tsne[idx, 0],
                    student_tsne[idx, 1],
                    c=colors[i],
                    label=f"Class {i}",
                    marker='*',
                    alpha=0.7
                )
        plt.title("Student Features (t-SNE)")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "feature_space_tsne.png"), dpi=150)
        plt.close()
    
    def visualize_attention_maps(self, features):
        """
        Visualize attention maps from teacher and student.
        
        Args:
            features: Dictionary with extracted features
        """
        if "teacher_attn" not in features or "student_attn" not in features:
            print("Attention maps not available for visualization")
            return
        
        # Get attention maps
        teacher_attn = features["teacher_attn"]
        student_attn = features["student_attn"]
        
        # Select a few samples
        num_samples = min(4, teacher_attn.shape[0])
        
        plt.figure(figsize=(15, 10))
        
        for i in range(num_samples):
            # Teacher attention map
            plt.subplot(2, num_samples, i + 1)
            sns.heatmap(
                teacher_attn[i],
                cmap='viridis',
                xticklabels=False,
                yticklabels=False
            )
            plt.title(f"Teacher Attention (Sample {i+1})")
            
            # Student attention map
            plt.subplot(2, num_samples, num_samples + i + 1)
            sns.heatmap(
                student_attn[i],
                cmap='viridis',
                xticklabels=False,
                yticklabels=False
            )
            plt.title(f"Student Attention (Sample {i+1})")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "attention_maps.png"), dpi=150)
        plt.close()
    
    def analyze_feature_correlations(self, features):
        """
        Analyze correlations between teacher and student features.
        
        Args:
            features: Dictionary with extracted features
        """
        # Get features
        teacher_skel_feat = features["teacher_skel_feat"]
        teacher_imu_feat = features["teacher_imu_feat"]
        teacher_fused_feat = features["teacher_fused_feat"]
        student_feat = features["student_feat"]
        
        # Calculate cosine similarity between different features
        def cosine_similarity(a, b):
            norm_a = np.linalg.norm(a, axis=1)
            norm_b = np.linalg.norm(b, axis=1)
            return np.sum(a * b, axis=1) / (norm_a * norm_b)
        
        # Calculate correlations
        correlations = {
            "Student-TeacherFused": np.mean(cosine_similarity(student_feat, teacher_fused_feat)),
            "Student-TeacherIMU": np.mean(cosine_similarity(student_feat, teacher_imu_feat)),
            "Student-TeacherSkel": np.mean(cosine_similarity(student_feat, teacher_skel_feat)),
            "TeacherIMU-TeacherSkel": np.mean(cosine_similarity(teacher_imu_feat, teacher_skel_feat)),
            "TeacherFused-TeacherIMU": np.mean(cosine_similarity(teacher_fused_feat, teacher_imu_feat)),
            "TeacherFused-TeacherSkel": np.mean(cosine_similarity(teacher_fused_feat, teacher_skel_feat))
        }
        
        # Create bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(correlations.keys(), correlations.values())
        plt.title("Feature Space Correlations")
        plt.xlabel("Feature Pair")
        plt.ylabel("Mean Cosine Similarity")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.output_dir, "feature_correlations.png"), dpi=150)
        plt.close()
        
        # Also save as text
        with open(os.path.join(self.output_dir, "feature_correlations.txt"), 'w') as f:
            for pair, corr in correlations.items():
                f.write(f"{pair}: {corr:.4f}\n")
        
        return correlations
    
    def create_summary(self, correlations):
        """
        Create summary report.
        
        Args:
            correlations: Dictionary with feature correlations
        """
        report_path = os.path.join(self.output_dir, "cross_modal_transfer_analysis.md")
        with open(report_path, 'w') as f:
            f.write("# Cross-Modal Knowledge Transfer Analysis\n\n")
            
            f.write("## Feature Space Analysis\n\n")
            f.write("This analysis examines how effectively knowledge from the skeleton modality has been transferred to the IMU-only student model.\n\n")
            
            f.write("### Feature Correlations\n\n")
            f.write("| Feature Pair | Cosine Similarity |\n")
            f.write("|--------------|------------------|\n")
            for pair, corr in correlations.items():
                f.write(f"| {pair} | {corr:.4f} |\n")
            
            f.write("\n### Interpretation\n\n")
            
            # Find strongest and weakest correlations
            strongest = max(correlations.items(), key=lambda x: x[1])
            weakest = min(correlations.items(), key=lambda x: x[1])
            
            # Calculate ratio of student-teacher correlations
            student_fused = correlations.get("Student-TeacherFused", 0)
            student_imu = correlations.get("Student-TeacherIMU", 0)
            student_skel = correlations.get("Student-TeacherSkel", 0)
            
            # Check if the student has learned to mimic the fused representation
            if student_fused > student_imu and student_fused > student_skel:
                transfer_quality = "strong"
            elif student_fused > student_skel:
                transfer_quality = "moderate"
            else:
                transfer_quality = "weak"
            
            f.write(f"- Strongest correlation: **{strongest[0]}** ({strongest[1]:.4f})\n")
            f.write(f"- Weakest correlation: **{weakest[0]}** ({weakest[1]:.4f})\n")
            f.write(f"- Cross-modal knowledge transfer quality: **{transfer_quality}**\n\n")
            
            if transfer_quality == "strong":
                f.write("The student model has successfully learned to mimic the teacher's fused representation, indicating effective cross-modal knowledge transfer from skeleton to IMU data.\n\n")
            elif transfer_quality == "moderate":
                f.write("The student model shows some ability to capture the teacher's fused representation, but there is room for improvement in cross-modal knowledge transfer.\n\n")
            else:
                f.write("The student model has not effectively captured the teacher's fused representation, suggesting limited cross-modal knowledge transfer.\n\n")
            
            # Check if student is closer to IMU or skeleton features
            if student_imu > student_skel:
                f.write("The student model is more aligned with the teacher's IMU features than the skeleton features, which is expected since the student only uses IMU data. However, the goal of cross-modal distillation is to incorporate skeleton knowledge into the IMU-only model.\n\n")
            else:
                f.write("Interestingly, the student model shows stronger alignment with the teacher's skeleton features than the IMU features, suggesting highly effective cross-modal knowledge transfer.\n\n")
            
            f.write("## Recommendations\n\n")
            
            if transfer_quality == "strong":
                f.write("1. **Fine-tune hyperparameters**: The current approach is working well but could be further optimized\n")
                f.write("2. **Explore other attention mechanisms**: Consider multi-head cross-attention between modalities in the teacher\n")
                f.write("3. **Investigate temporal knowledge**: Analyze how well temporal patterns from skeleton are captured in the student\n")
            elif transfer_quality == "moderate":
                f.write("1. **Adjust distillation weights**: Increase the weight of feature alignment loss\n")
                f.write("2. **Add intermediate feature alignment**: Align multiple intermediate layers, not just the final features\n")
                f.write("3. **Try contrastive learning**: Implement contrastive objectives to better align feature spaces\n")
            else:
                f.write("1. **Revise distillation approach**: The current method is not effectively transferring knowledge\n")
                f.write("2. **Add projector networks**: Use learned projections between teacher and student feature spaces\n")
                f.write("3. **Implement adversarial training**: Add a discriminator to push the student to better mimic the teacher\n")
                f.write("4. **Pre-train with reconstruction**: Have the student reconstruct skeleton data from IMU data as a pre-training step\n")
    
    def run_analysis(self):
        """Run complete analysis pipeline."""
        print("Loading models...")
        self.load_models()
        
        print("Preparing data...")
        data = self.prepare_data()
        
        print("Extracting features...")
        features = self.extract_features(data)
        
        print("Visualizing feature spaces...")
        self.visualize_feature_space(features)
        
        print("Visualizing attention maps...")
        self.visualize_attention_maps(features)
        
        print("Analyzing feature correlations...")
        correlations = self.analyze_feature_correlations(features)
        
        print("Creating summary report...")
        self.create_summary(correlations)
        
        print(f"Analysis completed. Results saved to {self.output_dir}")

def main():
    """Main function."""
    # Parse arguments
    arg = parse_args()
    
    # Run analysis
    visualizer = FeatureVisualizer(arg)
    visualizer.run_analysis()

if __name__ == "__main__":
    main()
