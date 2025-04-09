import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, args):
        super(LSTMModel, self).__init__()
        self.input_channels = args.input_channels
        self.hidden_channels = args.hidden_channels
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.num_classes = args.num_classes
        self.lstm = nn.LSTM(
            input_size=self.input_channels,
            hidden_size=self.hidden_channels,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=True
        )
        self.fusion_fc = nn.Linear(43, self.hidden_channels) if args.fusion and hasattr(args, 'fusion_features') and args.fusion_features else None
        self.fc = nn.Linear(self.hidden_channels * 2, self.num_classes)
        
    def forward(self, x):
        if isinstance(x, dict):
            if 'accelerometer' in x:
                x_acc = x['accelerometer']
                lstm_out, _ = self.lstm(x_acc)
                lstm_out = lstm_out[:, -1, :]
                if self.fusion_fc is not None and 'fusion_features' in x:
                    fusion_out = self.fusion_fc(x['fusion_features'])
                    lstm_out = lstm_out + fusion_out
                return self.fc(lstm_out)
            else:
                raise ValueError("Accelerometer data is required")
        else:
            lstm_out, _ = self.lstm(x)
            lstm_out = lstm_out[:, -1, :]
            return self.fc(lstm_out)

def get_model(args):
    if args.model.lower() == 'lstm':
        return LSTMModel(args)
    else:
        raise ValueError(f"Model {args.model} not recognized")import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, args):
        super(LSTMModel, self).__init__()
        self.input_channels = args.input_channels
        self.hidden_channels = args.hidden_channels
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.num_classes = args.num_classes
        self.lstm = nn.LSTM(
            input_size=self.input_channels,
            hidden_size=self.hidden_channels,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=True
        )
        self.fusion_fc = nn.Linear(43, self.hidden_channels) if args.fusion and hasattr(args, 'fusion_features') and args.fusion_features else None
        self.fc = nn.Linear(self.hidden_channels * 2, self.num_classes)
        
    def forward(self, x):
        if isinstance(x, dict):
            if 'accelerometer' in x:
                x_acc = x['accelerometer']
                lstm_out, _ = self.lstm(x_acc)
                lstm_out = lstm_out[:, -1, :]
                if self.fusion_fc is not None and 'fusion_features' in x:
                    fusion_out = self.fusion_fc(x['fusion_features'])
                    lstm_out = lstm_out + fusion_out
                return self.fc(lstm_out)
            else:
                raise ValueError("Accelerometer data is required")
        else:
            lstm_out, _ = self.lstm(x)
            lstm_out = lstm_out[:, -1, :]
            return self.fc(lstm_out)

def get_model(args):
    if args.model.lower() == 'lstm':
        return LSTMModel(args)
    else:
        raise ValueError(f"Model {args.model} not recognized")
