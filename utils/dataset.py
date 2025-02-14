# utils/dataset.py

import os
from typing import List, Dict
from utils.loader import DatasetBuilder

class ModalityFile:
    def __init__(self, subject_id, action_id, sequence_number, file_path):
        self.subject_id = subject_id
        self.action_id = action_id
        self.sequence_number = sequence_number
        self.file_path = file_path

class Modality:
    def __init__(self, name: str):
        self.name = name
        self.files: List[ModalityFile] = []

    def add_file(self, subject_id, action_id, seq, path):
        self.files.append(ModalityFile(subject_id, action_id, seq, path))

class MatchedTrial:
    def __init__(self, subject_id, action_id, sequence_number):
        self.subject_id = subject_id
        self.action_id = action_id
        self.sequence_number = sequence_number
        self.files: Dict[str, str] = {}

    def add_file(self, modality_name, file_path):
        self.files[modality_name] = file_path

class SmartFallMM:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.age_groups = {"old": {}, "young": {}}
        self.matched_trials: List[MatchedTrial] = []
        self.selected_sensors: Dict[str, str] = {}

    def add_modality(self, age_group, modality_name):
        self.age_groups[age_group][modality_name] = Modality(modality_name)

    def select_sensor(self, modality_name, sensor_name=None):
        if modality_name=='skeleton':
            self.selected_sensors[modality_name] = None
        else:
            if sensor_name is None:
                raise ValueError(f"Sensor required for {modality_name}")
            self.selected_sensors[modality_name] = sensor_name

    def load_files(self):
        import re
        for age_group, mod_dict in self.age_groups.items():
            for mod_name, mod_obj in mod_dict.items():
                if mod_name == 'skeleton':
                    base_dir = os.path.join(self.root_dir, age_group, mod_name)
                else:
                    sensor = self.selected_sensors.get(mod_name, None)
                    base_dir = os.path.join(self.root_dir, age_group, mod_name, sensor)
                if not os.path.isdir(base_dir):
                    continue

                for root, _, files in os.walk(base_dir):
                    for f in files:
                        if f.endswith('.csv'):
                            # parse SxxAxxTxx
                            # e.g. S43A03T01.csv
                            try:
                                subj = int(f[1:3])
                                act  = int(f[4:6])
                                seq  = int(f[7:9])
                                fp   = os.path.join(root, f)
                                mod_obj.add_file(subj, act, seq, fp)
                            except:
                                pass

    def match_trials(self):
        # Gather all trials
        trial_dict = {}
        # define required from 'young' as an example
        required_mods = list(self.age_groups['young'].keys())

        for age_group, mod_dict in self.age_groups.items():
            for mod_name, mod_obj in mod_dict.items():
                for mf in mod_obj.files:
                    key = (mf.subject_id, mf.action_id, mf.sequence_number)
                    if key not in trial_dict:
                        trial_dict[key] = {}
                    trial_dict[key][mod_name] = mf.file_path

        for (s, a, t), dmap in trial_dict.items():
            if all(rm in dmap for rm in required_mods):
                mt = MatchedTrial(s, a, t)
                for mk, path in dmap.items():
                    mt.add_file(mk, path)
                self.matched_trials.append(mt)

    def pipe_line(self, age_group, modalities, sensors):
        for ag in age_group:
            for mod in modalities:
                self.add_modality(ag, mod)
                if mod=='skeleton':
                    self.select_sensor('skeleton', None)
                else:
                    for sensor_name in sensors:
                        self.select_sensor(mod, sensor_name)
        self.load_files()
        self.match_trials()

SmartFallMM.pipe_line = SmartFallMM.pipe_line  # attach method for convenience

def prepare_smartfallmm(arg):
    """
    Build SmartFallMM dataset from local dir => 'data/smartfallmm', 
    run pipe_line => add modalities, sensors => match => return DatasetBuilder
    """
    import os
    from utils.loader import DatasetBuilder
    root_dir = os.path.join(os.getcwd(), 'data/smartfallmm')

    sf = SmartFallMM(root_dir)
    # e.g. arg.dataset_args = { 'age_group': ['young'], 'modalities': ['accelerometer','skeleton'], 'sensors':['watch'], ... }
    sf.pipe_line(
        age_group=arg.dataset_args.get('age_group', ['young']),
        modalities=arg.dataset_args.get('modalities', ['accelerometer']),
        sensors=arg.dataset_args.get('sensors', ['watch'])
    )
    builder = DatasetBuilder(
        dataset=sf,
        mode=arg.dataset_args.get('mode','variable_time'),
        max_length=arg.dataset_args.get('max_length',64),
        task=arg.dataset_args.get('task','fd'),
        window_size_sec=arg.dataset_args.get('window_size_sec',4.0),
        stride_sec=arg.dataset_args.get('stride_sec',0.5),
        time2vec_dim=arg.dataset_args.get('time2vec_dim',8)
    )
    return builder

