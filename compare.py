#!/usr/bin/env python
"""
Compare teacher and student model architectures for cross-modal distillation.
"""

import argparse
import yaml
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from importlib import import_module

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare teacher and student model architectures."
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/smartfallmm/distill_student_enhanced.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./architecture_analysis",
        help="Directory to save architecture analysis"
    )
    return parser.parse_args()

def count_parameters(model):
    """
    Count trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_layer_info(model):
    """
    Get information about model layers.
    
    Args:
        model: PyTorch model
        
    Returns:
        List of dictionaries with layer information
    """
    layers = []
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            layers.append({
                "name": name,
                "type": module.__class__.__name__,
                "params": params
            })
    
    return layers

def visualize_model_architecture(model, title, output_path):
    """
    Visualize model architecture as a bar chart.
    
    Args:
        model: PyTorch model
        title: Plot title
        output_path: Output file path
    """
    layers = get_layer_info(model)
    
    # Group by layer type
    layer_types = {}
    for layer in layers:
        layer_type = layer["type"]
        if layer_type not in layer_types:
            layer_types[layer_type] = 0
        layer_types[layer_type] += layer["params"]
    
    # Sort by parameter count
    sorted_types = sorted(layer_types.items(), key=lambda x: x[1], reverse=True)
    types = [t[0] for t in sorted_types]
    params = [t[1] for t in sorted_types]
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(types, params)
    
    # Add parameter counts on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f"{height/1000:.1f}K",
                 ha='center', va='bottom', rotation=45)
    
    plt.title(f"{title} - Parameters by Layer Type")
    plt.xlabel("Layer Type")
    plt.ylabel("Parameters")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150)
    plt.close()

def compare_architectures(teacher, student, output_dir):
    """
    Compare teacher and student architectures.
    
    Args:
        teacher: Teacher model
        student: Student model
        output_dir: Output directory
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Count parameters
    teacher_params = count_parameters(teacher)
    student_params = count_parameters(student)
    
    # Get layer info
    teacher_layers = get_layer_info(teacher)
    student_layers = get_layer_info(student)
    
    # Visualize architectures
    visualize_model_architecture(
        teacher, 
        "Teacher Model", 
        os.path.join(output_dir, "teacher_architecture.png")
    )
    
    visualize_model_architecture(
        student, 
        "Student Model", 
        os.path.join(output_dir, "student_architecture.png")
    )
    
    # Compare parameters
    plt.figure(figsize=(8, 6))
    plt.bar(["Teacher", "Student"], [teacher_params, student_params])
    plt.title("Model Size Comparison")
    plt.ylabel("Number of Parameters")
    
    # Add parameter counts
    for i, v in enumerate([teacher_params, student_params]):
        plt.text(i, v + 0.1, f"{v/1000:.1f}K", ha='center')
    
    # Add reduction percentage
    reduction = (1 - student_params / teacher_params) * 100
    plt.text(0.5, max(teacher_params, student_params) * 0.5,
             f"{reduction:.1f}% Reduction",
             ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_size_comparison.png"), dpi=150)
    plt.close()
    
    # Create summary report
    report_path = os.path.join(output_dir, "architecture_comparison.md")
    with open(report_path, 'w') as f:
        f.write("# Teacher-Student Architecture Comparison\n\n")
        
        f.write("## Model Size\n\n")
        f.write(f"- **Teacher Model**: {teacher_params:,} parameters\n")
        f.write(f"- **Student Model**: {student_params:,} parameters\n")
        f.write(f"- **Size Reduction**: {reduction:.1f}%\n\n")
        
        f.write("## Teacher Architecture\n\n")
        f.write("```\n")
        f.write(str(teacher))
        f.write("\n```\n\n")
        
        f.write("## Student Architecture\n\n")
        f.write("```\n")
        f.write(str(student))
        f.write("\n```\n\n")
        
        f.write("## Layer Analysis\n\n")
        
        f.write("### Teacher Layers\n\n")
        f.write("| Layer | Type | Parameters |\n")
        f.write("|-------|------|------------|\n")
        for layer in teacher_layers:
            f.write(f"| {layer['name']} | {layer['type']} | {layer['params']:,} |\n")
        
        f.write("\n### Student Layers\n\n")
        f.write("| Layer | Type | Parameters |\n")
        f.write("|-------|------|------------|\n")
        for layer in student_layers:
            f.write(f"| {layer['name']} | {layer['type']} | {layer['params']:,} |\n")
        
        f.write("\n## Key Differences\n\n")
        f.write("1. **Multi-Modal vs. Single-Modal**: Teacher processes both skeleton and IMU data, while student only processes IMU data\n")
        f.write("2. **Layer Depths**: Teacher has deeper transformer layers than student\n")
        f.write("3. **Hidden Dimensions**: Teacher uses larger hidden dimensions than student\n")
        f.write("4. **Fusion Strategy**: Teacher uses explicit fusion for skeleton and IMU features\n")

def main():
    """Main function."""
    # Parse arguments
    arg = parse_args()
    
    # Load configuration
    with open(arg.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Import model classes
    teacher_path = cfg["teacher_model"]
    student_path = cfg["student_model"]
    
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
    teacher = TeacherClass(**cfg["teacher_args"])
    student = StudentClass(**cfg["student_args"])
    
    # Compare architectures
    compare_architectures(teacher, student, arg.output_dir)
    
    print(f"Architecture comparison completed. Results saved to {arg.output_dir}")

if __name__ == "__main__":
    main()
