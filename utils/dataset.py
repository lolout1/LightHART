# utils/dataset.py
import os
import logging
import numpy as np
from utils.smart_fall_mm import SmartFallMMBuilder

logger = logging.getLogger(__name__)

def prepare_smartfallmm(args):
    logger.info("Preparing SmartFallMM dataset")

    if not hasattr(args, 'dataset_args') or not args.dataset_args:
        args.dataset_args = {
            'age_group': ['young', 'old'],
            'modalities': ['accelerometer'],
            'sensors': ['watch'],
            'mode': 'sliding_window',
            'max_length': 128,
            'task': 'fd'
        }

    root_dir = getattr(args, 'data_dir', 'data/smartfallmm')

    builder = SmartFallMMBuilder(
        root=root_dir,
        age_group=args.dataset_args['age_group'],
        modalities=args.dataset_args['modalities'],
        sensors=args.dataset_args['sensors'],
        mode=args.dataset_args['mode'],
        max_length=args.dataset_args['max_length'],
        task=args.dataset_args['task']
    )

    builder.run_pipeline()
    return builder

def split_by_subjects(builder, subjects, fuse=False):
    logger.info(f"Building dataset for subjects: {subjects}")
    try:
        return builder.build_dataset(subjects, fuse)
    except Exception as e:
        logger.error(f"Error building dataset: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'accelerometer': np.array([]),
            'labels': np.array([]),
            'subjects': np.array([])
        }
