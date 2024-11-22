from .loader import DatasetBuilder

def prepare_smartfallmm(arg) -> DatasetBuilder:
    """Prepares the SmartFallMM dataset for training or evaluation."""
    from .shared import SmartFallMM

    sm_dataset = SmartFallMM(
        root_dir=arg.dataset_args['root_dir'],
        age_groups=arg.dataset_args['age_group'],
        modalities=arg.dataset_args['modalities'],
        sensors=arg.dataset_args['sensors']
    )
    sm_dataset.load_files()
    return DatasetBuilder(
        dataset=sm_dataset,
        mode=arg.dataset_args['mode'],
        max_length=arg.dataset_args['max_length'],
        task=arg.dataset_args['task']
    )


def filter_subjects(builder, subjects):
    """Filters data for the given subjects."""
    builder.make_dataset(subjects)
    return builder.data
