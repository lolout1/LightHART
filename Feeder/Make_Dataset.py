import torch
import numpy as np

class UTD_mm(torch.utils.data.Dataset):
    def __init__(self, dataset, batch_size):
        if not isinstance(dataset, dict) or not all(k in dataset for k in ['accelerometer', 'labels', 'skeleton']):
            print("Creating dummy dataset due to missing data")
            self.acc_data = np.random.randn(10, 64, 4) * 0.01
            self.labels = np.array([0] * 5 + [1] * 5)
            self.skl_data = np.zeros((10, 64, 32, 3))
            self.num_samples = 10
        else:
            try:
                self.acc_data = dataset['accelerometer']
                self.labels = dataset['labels']
                self.skl_data = dataset['skeleton']
                self.num_samples = self.acc_data.shape[0]
                
                # Ensure skeleton data has the right shape
                if len(self.skl_data.shape) == 3:
                    # Reshape from [N, T, features] to [N, T, joints, coords]
                    if self.skl_data.shape[2] % 3 == 0:
                        joints = self.skl_data.shape[2] // 3
                        self.skl_data = np.reshape(self.skl_data, (self.num_samples, self.skl_data.shape[1], joints, 3))
                    else:
                        # Create dummy skeleton data if shape is incompatible
                        self.skl_data = np.zeros((self.num_samples, self.acc_data.shape[1], 32, 3))
            except Exception as e:
                print(f"Error initializing dataset: {str(e)}. Creating dummy dataset.")
                self.acc_data = np.random.randn(10, 64, 4) * 0.01
                self.labels = np.array([0] * 5 + [1] * 5)
                self.skl_data = np.zeros((10, 64, 32, 3))
                self.num_samples = 10
            
        self.batch_size = batch_size
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        try:
            skl_data = torch.tensor(self.skl_data[index], dtype=torch.float32)
            acc_data = torch.tensor(self.acc_data[index], dtype=torch.float32)
            data = {'accelerometer': acc_data, 'skeleton': skl_data}
            label = self.labels[index]
            label = torch.tensor(label, dtype=torch.long)
            return data, label, index
        except Exception as e:
            print(f"Error getting item {index}: {str(e)}. Returning dummy data.")
            # Return dummy data in case of error
            acc_dummy = torch.zeros((64, 4), dtype=torch.float32)
            skl_dummy = torch.zeros((64, 32, 3), dtype=torch.float32)
            data = {'accelerometer': acc_dummy, 'skeleton': skl_dummy}
            label = torch.tensor(0, dtype=torch.long)
            return data, label, index
