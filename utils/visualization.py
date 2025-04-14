# utils/visualization.py
import matplotlib.pyplot as plt
import numpy as np
import os

class AccelerometerVisualizer:
    def __init__(self, output_dir='visualizations'):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
    
    def visualize_raw_data(self, timestamps, values, filename='raw_data.png'):
        plt.figure(figsize=(12, 6))
        seconds = [(t - timestamps[0]).total_seconds() for t in timestamps]
        
        # Plot raw data
        plt.subplot(2, 1, 1)
        plt.plot(seconds, values[:, 0], 'r-', label='X-axis')
        plt.plot(seconds, values[:, 1], 'g-', label='Y-axis')
        plt.plot(seconds, values[:, 2], 'b-', label='Z-axis')
        plt.title('Raw Accelerometer Data')
        plt.ylabel('Acceleration (m/s²)')
        plt.legend()
        
        # Plot sampling intervals
        plt.subplot(2, 1, 2)
        intervals = np.diff(seconds)
        plt.plot(seconds[1:], intervals, 'k.')
        plt.title('Sampling Intervals')
        plt.xlabel('Time (s)')
        plt.ylabel('Interval (s)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
    
    def visualize_resampling(self, original_seconds, original_values, resampled_seconds, resampled_values, filename='resampling.png'):
        plt.figure(figsize=(12, 8))
        
        # Plot X-axis
        plt.subplot(3, 1, 1)
        plt.plot(original_seconds, original_values[:, 0], 'r.', alpha=0.5, label='Original')
        plt.plot(resampled_seconds, resampled_values[:, 0], 'r-', label='Resampled')
        plt.title('X-axis Acceleration')
        plt.ylabel('Acceleration (m/s²)')
        plt.legend()
        
        # Plot Y-axis
        plt.subplot(3, 1, 2)
        plt.plot(original_seconds, original_values[:, 1], 'g.', alpha=0.5, label='Original')
        plt.plot(resampled_seconds, resampled_values[:, 1], 'g-', label='Resampled')
        plt.title('Y-axis Acceleration')
        plt.ylabel('Acceleration (m/s²)')
        plt.legend()
        
        # Plot Z-axis
        plt.subplot(3, 1, 3)
        plt.plot(original_seconds, original_values[:, 2], 'b.', alpha=0.5, label='Original')
        plt.plot(resampled_seconds, resampled_values[:, 2], 'b-', label='Resampled')
        plt.title('Z-axis Acceleration')
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration (m/s²)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
    
    def visualize_filtering(self, time_points, unfiltered_data, filtered_data, filename='filtering.png'):
        plt.figure(figsize=(12, 8))
        
        # Plot X-axis
        plt.subplot(3, 1, 1)
        plt.plot(time_points, unfiltered_data[:, 0], 'r-', alpha=0.5, label='Unfiltered')
        plt.plot(time_points, filtered_data[:, 0], 'r-', label='Filtered')
        plt.title('X-axis: Butterworth Filter Comparison')
        plt.ylabel('Acceleration (m/s²)')
        plt.legend()
        
        # Plot Y-axis
        plt.subplot(3, 1, 2)
        plt.plot(time_points, unfiltered_data[:, 1], 'g-', alpha=0.5, label='Unfiltered')
        plt.plot(time_points, filtered_data[:, 1], 'g-', label='Filtered')
        plt.title('Y-axis: Butterworth Filter Comparison')
        plt.ylabel('Acceleration (m/s²)')
        plt.legend()
        
        # Plot Z-axis
        plt.subplot(3, 1, 3)
        plt.plot(time_points, unfiltered_data[:, 2], 'b-', alpha=0.5, label='Unfiltered')
        plt.plot(time_points, filtered_data[:, 2], 'b-', label='Filtered')
        plt.title('Z-axis: Butterworth Filter Comparison')
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration (m/s²)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
    
    def visualize_normalization(self, time_points, raw_data, normalized_data, filename='normalization.png'):
        plt.figure(figsize=(12, 8))
        
        # Plot X-axis
        plt.subplot(3, 1, 1)
        plt.plot(time_points, raw_data[:, 0], 'r-', alpha=0.5, label='Raw')
        plt.plot(time_points, normalized_data[:, 0], 'r-', label='Normalized')
        plt.title('X-axis: Normalization Comparison')
        plt.ylabel('Acceleration')
        plt.legend()
        
        # Plot Y-axis
        plt.subplot(3, 1, 2)
        plt.plot(time_points, raw_data[:, 1], 'g-', alpha=0.5, label='Raw')
        plt.plot(time_points, normalized_data[:, 1], 'g-', label='Normalized')
        plt.title('Y-axis: Normalization Comparison')
        plt.ylabel('Acceleration')
        plt.legend()
        
        # Plot Z-axis
        plt.subplot(3, 1, 3)
        plt.plot(time_points, raw_data[:, 2], 'b-', alpha=0.5, label='Raw')
        plt.plot(time_points, normalized_data[:, 2], 'b-', label='Normalized')
        plt.title('Z-axis: Normalization Comparison')
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
    
    def compare_resampling_methods(self, original_seconds, original_values, methods_dict, filename='resampling_methods.png'):
        plt.figure(figsize=(15, 10))
        
        # Prepare resampled seconds (assuming all methods have the same length)
        first_method = list(methods_dict.values())[0]
        duration = original_seconds[-1]
        num_points = first_method.shape[0]
        resampled_seconds = np.linspace(0, duration, num_points)
        
        # Plot X-axis comparison
        plt.subplot(3, 1, 1)
        plt.plot(original_seconds, original_values[:, 0], 'k.', alpha=0.3, label='Original')
        for method_name, resampled_data in methods_dict.items():
            plt.plot(resampled_seconds, resampled_data[:, 0], label=method_name)
        plt.title('X-axis: Resampling Method Comparison')
        plt.ylabel('Acceleration (m/s²)')
        plt.legend()
        
        # Plot Y-axis comparison
        plt.subplot(3, 1, 2)
        plt.plot(original_seconds, original_values[:, 1], 'k.', alpha=0.3, label='Original')
        for method_name, resampled_data in methods_dict.items():
            plt.plot(resampled_seconds, resampled_data[:, 1], label=method_name)
        plt.title('Y-axis: Resampling Method Comparison')
        plt.ylabel('Acceleration (m/s²)')
        plt.legend()
        
        # Plot Z-axis comparison
        plt.subplot(3, 1, 3)
        plt.plot(original_seconds, original_values[:, 2], 'k.', alpha=0.3, label='Original')
        for method_name, resampled_data in methods_dict.items():
            plt.plot(resampled_seconds, resampled_data[:, 2], label=method_name)
        plt.title('Z-axis: Resampling Method Comparison')
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration (m/s²)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
    
    def visualize_spectral_content(self, data, sampling_rate, before_filter=None, after_filter=None, filename='spectral_content.png'):
        plt.figure(figsize=(15, 10))
        
        # Compute FFT
        def compute_fft(signal):
            n = len(signal)
            freq = np.fft.rfftfreq(n, d=1/sampling_rate)
            fft_vals = np.fft.rfft(signal)
            magnitude = np.abs(fft_vals) / n
            return freq, magnitude
        
        axes_labels = ['X-axis', 'Y-axis', 'Z-axis']
        
        for i in range(3):
            plt.subplot(3, 1, i+1)
            
            # If before/after filtering data is provided
            if before_filter is not None and after_filter is not None:
                freq_before, mag_before = compute_fft(before_filter[:, i])
                freq_after, mag_after = compute_fft(after_filter[:, i])
                
                plt.plot(freq_before, mag_before, 'r-', alpha=0.5, label='Before Filter')
                plt.plot(freq_after, mag_after, 'b-', label='After Filter')
                plt.title(f'{axes_labels[i]}: Spectral Content Before/After Filtering')
            else:
                # Just show the spectrum of the data
                freq, mag = compute_fft(data[:, i])
                plt.plot(freq, mag, 'g-', label='Magnitude')
                plt.title(f'{axes_labels[i]}: Spectral Content')
            
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude')
            plt.legend()
            
            # Mark the cutoff frequency if filtering comparison
            if before_filter is not None and after_filter is not None:
                cutoff = 7.5  # The Butterworth cutoff
                plt.axvline(x=cutoff, color='k', linestyle='--', label=f'Cutoff ({cutoff} Hz)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
    
    def visualize_full_pipeline(self, raw_data, resampled_data, filtered_data, normalized_data, filename='full_pipeline.png'):
        """Visualize the complete preprocessing pipeline for a segment of data"""
        # Take a segment of data for clarity (e.g., first 500 points)
        segment_length = min(500, len(normalized_data))
        
        plt.figure(figsize=(15, 12))
        
        # Select a segment from each processing stage
        raw_segment = raw_data[:segment_length] if len(raw_data) >= segment_length else raw_data
        resampled_segment = resampled_data[:segment_length]
        filtered_segment = filtered_data[:segment_length]
        normalized_segment = normalized_data[:segment_length]
        
        # Time points
        time_points = np.arange(segment_length) / 25  # Assuming 25 Hz
        
        # For each axis
        axes_labels = ['X-axis', 'Y-axis', 'Z-axis']
        for i in range(3):
            plt.subplot(3, 1, i+1)
            
            # If raw data is available and of sufficient length
            if len(raw_segment) > 0:
                raw_time = np.linspace(0, time_points[-1], len(raw_segment))
                plt.plot(raw_time, raw_segment[:, i], 'k.', alpha=0.3, label='Raw')
            
            plt.plot(time_points, resampled_segment[:, i], 'r-', alpha=0.5, label='Resampled')
            plt.plot(time_points, filtered_segment[:, i], 'g-', alpha=0.5, label='Filtered')
            plt.plot(time_points, normalized_segment[:, i], 'b-', label='Normalized')
            
            plt.title(f'{axes_labels[i]}: Full Preprocessing Pipeline')
            plt.xlabel('Time (s)')
            plt.ylabel('Acceleration')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
