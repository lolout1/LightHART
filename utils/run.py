import os
import matplotlib.pyplot as plt
from dataset import SmartFallMM
from loader import DatasetBuilder

def visualize_and_save(builder, output_dir, subjects):
    """
    Visualize and save graphs for every trial for each subject in the range.
    
    Args:
        builder: DatasetBuilder object
        output_dir: Directory where graphs will be saved
        subjects: List of subjects to visualize
    """
    print(f"\n[INFO] Saving visualizations to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    for trial_index, trial in enumerate(builder.dataset.matched_trials):
        if trial.subject_id in subjects:
            trial_label = f"S{trial.subject_id:02d}A{trial.action_id:02d}T{trial.sequence_number:02d}"
            print(f"[INFO] Visualizing trial {trial_label}...")

            for modality_key in ["accelerometer_phone", "accelerometer_watch"]:
                if modality_key in trial.files:
                    try:
                        # Generate visualization
                        fig, axes = builder.visualize_trial(trial_index, modality_key, save_fig=False)

                        # Save figure to file
                        file_name = f"{trial_label}_{modality_key}.png"
                        file_path = os.path.join(output_dir, file_name)
                        fig.savefig(file_path, dpi=150)
                        plt.close(fig)  # Close the figure to free memory
                        print(f"[INFO] Saved {file_name}")
                    except Exception as e:
                        print(f"[ERROR] Could not visualize {trial_label} ({modality_key}): {e}")

def main():
    # Define the root directory of your dataset
    data_root = os.path.join(os.getcwd(), "data", "smartfallmm")
    output_dir = os.path.join(os.getcwd(), "visualizations")

    # Initialize the dataset
    dataset = SmartFallMM(root_dir=data_root)

    # Add accelerometer modalities for young and old groups
    dataset.add_modality("young", "accelerometer", ["phone", "watch"])
    dataset.add_modality("old", "accelerometer", ["phone", "watch"])

    # Load files and match trials
    print("\n[INFO] Loading and matching files...")
    dataset.load_files()
    dataset.match_trials()

    # Initialize DatasetBuilder
    builder = DatasetBuilder(
        dataset=dataset,
        mode="avg_pool",   # Can be 'avg_pool' or 'sliding_window'
        max_length=256,    # Adjust based on your requirements
        task="fd"          # Task type: 'fd' for fall detection
    )

    # Select subjects for the dataset (subjects 29 through 46)
    subjects = list(range(29, 47))  # Inclusive range: 29 to 46
    print(f"\n[INFO] Preparing dataset for subjects: {subjects}")
    builder.make_dataset(subjects=subjects)

    # Normalize the dataset
    print("\n[INFO] Normalizing dataset...")
    builder.normalization()

    # Visualize and save graphs
    visualize_and_save(builder, output_dir, subjects)

if __name__ == "__main__":
    main()
