import numpy as np
import torch
import logging
from torch.utils.data import Dataset

logger = logging.getLogger("feeder")

class UTD_mm(Dataset):
    def __init__(self, dataset=None, **kwargs):
        self.dataset = dataset
        self.data_acc = None
        self.data_gyro = None
        self.data_quat = None
        self.data_label = None
        self.subject_ids = None
        self.data_length = 0
        
        if dataset is not None:
            self.prepare_data(dataset)
    
    def prepare_data(self, dataset):
        self.data_acc = dataset.get('accelerometer', None)
        self.data_gyro = dataset.get('gyroscope', None)
        self.data_quat = dataset.get('quaternion', None)
        self.data_label = dataset.get('labels', None)
        self.subject_ids = dataset.get('subject_ids', None)
        
        if self.data_acc is not None:
            self.data_length = len(self.data_acc)
    
    def __len__(self):
        return self.data_length
    
    def __getitem__(self, index):
        if self.data_acc is None:
            logger.error(f"No data loaded at index {index}")
            return np.zeros((64, 3)), 0, 0
        
        data_dict = {}
        
        if self.data_acc is not None:
            data_dict['accelerometer'] = self.data_acc[index]
        
        if self.data_gyro is not None:
            data_dict['gyroscope'] = self.data_gyro[index]
        
        if self.data_quat is not None:
            data_dict['quaternion'] = self.data_quat[index]
        
        label = self.data_label[index] if self.data_label is not None else 0
        subject_id = self.subject_ids[index] if self.subject_ids is not None else 0
        
        return data_dict, label, subject_id

class UTD_mm_RealTime(Dataset):
    def __init__(self, window_size=64, stride=10, filter_type='madgwick', **kwargs):
        self.window_size = window_size
        self.stride = stride
        self.filter_type = filter_type
        self.current_filter = None
        self.reset()
    
    def reset(self):
        self.acc_buffer = np.zeros((self.window_size, 3))
        self.gyro_buffer = np.zeros((self.window_size, 3))
        self.quat_buffer = np.zeros((self.window_size, 4))
        self.timestamps = np.zeros(self.window_size)
        self.current_idx = 0
        self.total_samples = 0
        self.buffer_full = False
        self.windows_generated = 0
        
        self.current_filter.reset()
    
    def add_sample(self, acc, gyro, timestamp=None):
        if timestamp is None:
            timestamp = self.total_samples / 30.0  # Assume 30Hz if no timestamp
        
        self.total_samples += 1
        
        if self.buffer_full:
            # Shift the buffer by stride if we've accumulated enough samples
            if self.current_idx >= self.stride:
                self.acc_buffer[:-self.stride] = self.acc_buffer[self.stride:]
                self.gyro_buffer[:-self.stride] = self.gyro_buffer[self.stride:]
                self.quat_buffer[:-self.stride] = self.quat_buffer[self.stride:]
                self.timestamps[:-self.stride] = self.timestamps[self.stride:]
                
                # Zero out the stride region
                self.acc_buffer[-self.stride:] = 0
                self.gyro_buffer[-self.stride:] = 0
                self.quat_buffer[-self.stride:] = 0
                self.timestamps[-self.stride:] = 0
                
                self.current_idx = self.window_size - self.stride
                self.windows_generated += 1
            
            # Add new sample
            self.acc_buffer[self.current_idx] = acc
            self.gyro_buffer[self.current_idx] = gyro
            self.timestamps[self.current_idx] = timestamp
            
            # Update quaternion (maintaining filter state)
            quat = self.current_filter.update(acc, gyro, timestamp)
            self.quat_buffer[self.current_idx] = quat
            
            self.current_idx += 1
        else:
            # Fill buffer sequentially until full
            self.acc_buffer[self.current_idx] = acc
            self.gyro_buffer[self.current_idx] = gyro
            self.timestamps[self.current_idx] = timestamp
            
            # Update quaternion (maintaining filter state)
            quat = self.current_filter.update(acc, gyro, timestamp)
            self.quat_buffer[self.current_idx] = quat
            
            self.current_idx += 1
            
            if self.current_idx >= self.window_size:
                self.buffer_full = True
                self.windows_generated += 1
    
    def get_current_window(self):
        if not self.buffer_full and self.current_idx == 0:
            return None
        
        # Return full window with all available samples
        valid_samples = self.window_size if self.buffer_full else self.current_idx
        
        data_dict = {
            'accelerometer': self.acc_buffer[:valid_samples].copy(),
            'gyroscope': self.gyro_buffer[:valid_samples].copy(),
            'quaternion': self.quat_buffer[:valid_samples].copy(),
            'timestamps': self.timestamps[:valid_samples].copy()
        }
        
        # If buffer isn't full but we have some data, pad to full window size
        if not self.buffer_full and self.current_idx > 0:
            for key in ['accelerometer', 'gyroscope', 'quaternion']:
                pad_shape = list(data_dict[key].shape)
                pad_shape[0] = self.window_size
                padded = np.zeros(pad_shape)
                padded[:self.current_idx] = data_dict[key]
                data_dict[key] = padded
            
            # Also pad timestamps
            padded_timestamps = np.zeros(self.window_size)
            padded_timestamps[:self.current_idx] = data_dict['timestamps']
            data_dict['timestamps'] = padded_timestamps
        
        return data_dict
    
    def should_process(self):
        # Return true if we have a full window or enough samples for inference
        return self.buffer_full or (self.current_idx >= self.window_size // 2)
    
    def get_windows_generated(self):
        return self.windows_generated
    
    def __len__(self):
        return 1 if self.should_process() else 0
    
    def __getitem__(self, index):
        window = self.get_current_window()
        if window is None:
            return {'accelerometer': np.zeros((self.window_size, 3))}, 0, 0
        
        return window, 0, 0  # No label or subject_id in real-time mode
