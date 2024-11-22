from typing import Any, List
from abc import ABC, abstractmethod
from scipy.io import loadmat
import pandas as pd
import numpy as np 
import torch
import torch.nn.functional as F

def csvloader(file_path: str, **kwargs):
    '''
    Loads CSV data, handling different formats between 'young' and 'old' datasets.
    '''
    try:
        # Read the CSV file without specifying header
        file_data = pd.read_csv(file_path, index_col=False, header=None, skip_blank_lines=True).dropna().bfill()
        num_columns = file_data.shape[1]

        # Assign default column names
        file_data.columns = [f'Column_{i}' for i in range(num_columns)]

        # Check if the file has a header by examining the first row
        first_row = file_data.iloc[0]
        if any('time' in str(item).lower() or 'timestamp' in str(item).lower() for item in first_row):
            # The file has a header
            file_data = pd.read_csv(file_path, index_col=False, header=0, skip_blank_lines=True).dropna().bfill()
        else:
            # The file does not have a header
            # We have already assigned default column names

            # If the first column is an index (monotonic increasing integers), drop it
            first_column = file_data.iloc[:, 0]
            if first_column.is_monotonic_increasing and pd.api.types.is_integer_dtype(first_column):
                file_data = file_data.drop(file_data.columns[0], axis=1)

        # Reassign column names after dropping index column
        num_columns = file_data.shape[1]
        file_data.columns = [f'Column_{i}' for i in range(num_columns)]

        # Identify timestamp columns
        timestamp_cols = [col for col in file_data.columns if 'time' in col.lower() or 'timestamp' in col.lower()]

        # If no timestamp columns are found, assume the first column is the timestamp
        if not timestamp_cols:
            timestamp_cols = [file_data.columns[0]]

        # Drop timestamp columns
        data_columns = [col for col in file_data.columns if col not in timestamp_cols]
        print(f"Data types before conversion:\n{file_data[data_columns].dtypes}")
        # Convert numerical columns to float32
        numerical_data = file_data[data_columns].astype(np.float32)

        activity_data = numerical_data.to_numpy()
        print(f"File: {file_path}")
        print(f"Columns: {file_data.columns}")
        print(f"Data sample:\n{file_data.head()}")
        if activity_data.size == 0:
            print(f"Warning: No data extracted from {file_path}.")
    
        return activity_data
    
    except Exception as e:
        print(f"Error loading CSV file {file_path}: {e}")
        raise
    

def matloader(file_path: str, **kwargs):
    '''
    Loads MatLab files 
    '''
    key = kwargs.get('key',None)
    assert key in ['d_iner' , 'd_skel'] , f'Unsupported {key} for matlab file'
    data = loadmat(file_path)[key]
    return data

LOADER_MAP = {
    'csv' : csvloader, 
    'mat' : matloader
}

def avg_pool(sequence : np.array, window_size : int = 5, stride :int =1, 
             max_length : int = 512 , shape : int = None) -> np.ndarray:

    '''
    Executes average pooling to smoothen out the data

    '''
    shape = sequence.shape
    sequence = sequence.reshape(shape[0], -1)
    sequence = np.expand_dims(sequence, axis = 0).transpose(0,2, 1)
    sequence = torch.tensor(sequence, dtype=torch.float32)
    stride =  ((sequence.shape[2]//max_length)+1 if max_length < sequence.shape[2] else 1)
    sequence = F.avg_pool1d(sequence,kernel_size=window_size, stride=stride)
    sequence = sequence.squeeze(0).numpy().transpose(1,0)
    sequence = sequence.reshape(-1, *shape[1:])
    return sequence


def pad_sequence_numpy(sequence: np.ndarray, max_sequence_length: int, 
                       input_shape: np.array) -> np.ndarray:
    '''
    Pools and pads the sequence to uniform length

    Args:
        sequence : data 
        max_sequence_length(int) : the fixed length of data
        input_shape: shape of the data
    Return: 
        new_sequence: data after padding
    '''
    shape = list(input_shape)
    shape[0] = max_sequence_length
    pooled_sequence = avg_pool(sequence=sequence, max_length = max_sequence_length, shape = input_shape)
    new_sequence = np.zeros(shape, sequence.dtype)
    new_sequence[:len(pooled_sequence)] = pooled_sequence
    return new_sequence

def sliding_window(data : np.ndarray, clearing_time_index : int, max_time : int, 
                   sub_window_size : int, stride_size : int) -> np.ndarray:
    '''
    Sliding Window
    '''

    assert clearing_time_index >= sub_window_size - 1 , "Clearing value needs to be greater or equal to (window size - 1)"
    start = clearing_time_index - sub_window_size + 1 

    if max_time >= data.shape[0]-sub_window_size:
        max_time = max_time - sub_window_size + 1
        # 2510 // 100 - 1 25 #25999 1000 24000 = 24900

    sub_windows  = (
        start + 
        np.expand_dims(np.arange(sub_window_size), 0) + 
        np.expand_dims(np.arange(max_time, step = stride_size), 0).T
    )

    #labels = np.round(np.mean(labels[sub_windows], axis=1))
    return data[sub_windows]


class Processor(ABC):
    '''
    Data Processor 
    '''
    def __init__(self, file_path:str, mode : str, max_length: str, **kwargs):
        assert mode in ['sliding_window', 'avg_pool'], f'Processing mode: {mode} is undefined'
        self.mode = mode
        self.max_length = max_length
        self.data = []
        self.file_path = file_path
        self.input_shape = []
        self.kwargs = kwargs


    def _set_input_shape(self, sequence: np.ndarray) -> List[int]:
        '''
        returns the shape of the inputj

        Args: 
            sequence(np.ndarray) : data sequence
        
        Out: 
            shape (list) : shape of the sequence
        '''
        self.input_shape =  sequence.shape


    def _import_loader(self, file_path:str) -> np.array :
        '''
        Reads file and loads data from
         
        '''

        file_type = file_path.split('.')[-1]

        assert file_type in ['csv', 'mat'], f'Unsupported file type {file_type}'

        return LOADER_MAP[file_type]

    def process(self):
        '''
        function implementation to process data
        '''
        loader = self._import_loader(self.file_path)
        data = loader(self.file_path, **self.kwargs)
        self._set_input_shape(data)
        if self.mode == 'avg_pool':
            data = pad_sequence_numpy(sequence=data, max_sequence_length=self.max_length,
                                      input_shape=self.input_shape)
        
        else: 
            data = sliding_window(data=data, clearing_time_index=self.max_length-1, 
                                  max_time=self.input_shape[0],
                                   sub_window_size =self.max_length, stride_size=1)
        return data

            