import torch
import os
import logging
import numpy as np
import json

logger = logging.getLogger("converter")

def export_to_torchscript(model_path, output_path, model_metadata_path=None):
    from models.fall_detection import FallDetectionTransformer
    
    if model_metadata_path and os.path.exists(model_metadata_path):
        with open(model_metadata_path, 'r') as f:
            metadata = json.load(f)
        model_params = metadata.get('model_parameters', {})
        input_shape = metadata.get('input_shape', [1, 128, 4])
    else:
        model_params = {
            'num_classes': 1,
            'num_heads': 4,
            'num_layer': 2,
            'embed_dim': 32
        }
        input_shape = [1, 128, 4]
    
    model = FallDetectionTransformer(
        acc_frames=128,
        num_classes=model_params.get('num_classes', 1),
        num_heads=model_params.get('num_heads', 4),
        acc_coords=3,
        num_layer=model_params.get('num_layer', 2),
        embed_dim=model_params.get('embed_dim', 32)
    )
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    dummy_input = torch.randn(tuple(input_shape))
    
    try:
        traced_model = torch.jit.trace(model, dummy_input)
        torch.jit.save(traced_model, output_path)
        logger.info(f"Model exported successfully to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to export model: {e}")
        return False
