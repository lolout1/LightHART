import os
import numpy as np
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
import time
import json
import pickle

from utils.processor.base import Processor, robust_align_modalities
from utils.imu_fusion import (
    StandardKalmanIMU, ExtendedKalmanIMU, UnscentedKalmanIMU,
    calibrate_filter, extract_orientation_from_skeleton
)

###############################################################################
# Combined "Approach A + B" in a single DatasetBuilder
###############################################################################

class DatasetBuilder:
    """
    Hybrid multi-threaded dataset builder that:
      1) Calibrates filter parameters from skeleton if phone & watch & skeleton 
         are present in representative trials (Approach A style).
      2) Uses caching for processed data, advanced alignment, robust skipping, 
         partial phone fallback (Approach B).
      3) If phone data is missing or partial, we fallback to watch-only logic, 
         skipping advanced phone-based fusion. 
      4) If watch is missing => skip trial entirely. (No watch => no inference).
    """

    def __init__(
        self, 
        dataset, 
        mode, 
        max_length, 
        task='fd',
        window_size_sec=4.0, 
        stride_sec=0.5, 
        imu_fusion='ekf',   # 'standard','ekf','ukf'
        align_method='dtw', # 'dtw','interpolation','crop'
        **kwargs
    ):
        """
        Args:
          dataset: e.g. your SmartFallMM or comparable object with matched_trials
          mode: e.g. 'variable_time'
          max_length: used if there's any length constraints
          task: 'fd' or 'age' or custom
          window_size_sec: typically 4.0
          stride_sec: typically 0.5 or 1.0
          imu_fusion: 'standard','ekf','ukf' => chosen filter
          align_method: how to align skeleton & IMU (e.g. 'dtw')
          kwargs: extra config 
            - calibrate_filter (bool) => do calibration
            - calibration_samples (int) => #representative trials
            - cache_dir => str path
            - use_cache => bool
            - skel_error_strategy => 'drop_trial' or 'keep_partial'
            - etc.
        """
        self.dataset = dataset
        self.mode = mode
        self.max_length = max_length
        self.task = task
        self.window_size_sec = window_size_sec
        self.stride_sec = stride_sec
        self.imu_fusion = imu_fusion
        self.align_method = align_method
        self.kwargs = kwargs

        # We store filter params (process_noise, measurement_noise, gyro_bias_noise)
        # Possibly from calibration
        self.filter_params = {
            'process_noise': 0.01,
            'measurement_noise': 0.1,
            'gyro_bias_noise': 0.01
        }
        self.calibrated = False

        self.do_calibration = kwargs.get('calibrate_filter', True)
        self.calibration_samples = kwargs.get('calibration_samples', 5)

        # Caching directories
        self.cache_dir = kwargs.get('cache_dir', './.cache')
        self.use_cache = kwargs.get('use_cache', True)
        os.makedirs(self.cache_dir, exist_ok=True)

        # Additional calibration cache
        self.calibration_cache_dir = os.path.join(self.cache_dir, 'calibration')
        os.makedirs(self.calibration_cache_dir, exist_ok=True)

        # Skeleton error strategy
        self.skel_error_strategy = kwargs.get('skel_error_strategy','drop_trial')

        # Final data
        self.data = {}
        self.processed_data = {'labels': []}

    ############################################################################
    # Basic Label / File Functions
    ############################################################################
    def _trial_label(self, trial):
        """Compute label from trial based on self.task."""
        if self.task == 'fd':
            return int(trial.action_id > 9)
        elif self.task == 'age':
            return int(trial.subject_id < 29 or trial.subject_id > 46)
        else:
            return trial.action_id - 1

    def get_cache_filename(self, trial, subjects):
        """Generate a unique cache filename for each trial + subject combo."""
        subject_str = '_'.join(map(str, subjects))
        return os.path.join(
            self.cache_dir, 
            f"s{trial.subject_id}_a{trial.action_id}_t{trial.sequence_number}_subs{subject_str}_fuse{self.imu_fusion}_align{self.align_method}.npz"
        )

    def get_calibration_cache_filename(self):
        """Where we store the calibrations for the chosen filter."""
        return os.path.join(
            self.calibration_cache_dir,
            f"kalman_params_{self.imu_fusion}.json"
        )

    ############################################################################
    # Filter Calibration Methods
    ############################################################################
    def load_calibration_parameters(self):
        """Attempt to load pre-calibrated filter params from .json cache."""
        path = self.get_calibration_cache_filename()
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    params = json.load(f)
                self.filter_params = params
                self.calibrated = True
                print(f"DEBUG: Loaded calibrated {self.imu_fusion} filter params => {params}")
                return True
            except Exception as e:
                print(f"ERROR: loading calibration param from {path}: {e}")
        return False

    def save_calibration_parameters(self):
        """Save the newly found filter params to .json."""
        path = self.get_calibration_cache_filename()
        try:
            with open(path, 'w') as f:
                json.dump(self.filter_params, f)
            print(f"DEBUG: Saved calibrated filter parameters to {path}")
        except Exception as e:
            print(f"ERROR: saving calibration parameters: {e}")

    def get_representative_trials(self, subjects, n_samples=5):
        """
        Balanced sampling of fall vs. non-fall to calibrate filter
        (like Approach A). 
        """
        fall_trials = []
        nonfall_trials = []

        for trial in self.dataset.matched_trials:
            if trial.subject_id not in subjects:
                continue
            # Must have watch accel, watch gyro, skeleton (phone optional but let's prefer phone if we want advanced calibration)
            modkeys = trial.files.keys()
            if ('accelerometer' not in modkeys) or ('gyroscope' not in modkeys) or ('skeleton' not in modkeys):
                continue
            label = self._trial_label(trial)
            if label == 1:  # fall
                fall_trials.append(trial)
            else:
                nonfall_trials.append(trial)

        np.random.shuffle(fall_trials)
        np.random.shuffle(nonfall_trials)

        n_fall = min(n_samples//2, len(fall_trials))
        n_nonfall = min(n_samples - n_fall, len(nonfall_trials))
        if n_fall == 0 and len(fall_trials)>0:
            n_fall = min(n_samples, len(fall_trials))
        if n_nonfall == 0 and len(nonfall_trials)>0:
            n_nonfall = min(n_samples, len(nonfall_trials))

        selected = fall_trials[:n_fall] + nonfall_trials[:n_nonfall]
        np.random.shuffle(selected)
        return selected

    def calibrate_filter_parameters(self, subjects):
        """
        Using Approach A's style: 
         - tries to load from .json 
         - if not found, picks representative trials with watch accel+gyro + skeleton, 
           uses them to calibrate. 
        """
        if self.load_calibration_parameters():
            return  # already loaded

        print(f"DEBUG: Attempting to calibrate {self.imu_fusion} filter params..")
        reps = self.get_representative_trials(subjects, self.calibration_samples)
        if not reps:
            print("DEBUG: No suitable calibration trials found. Using defaults.")
            return

        all_params = []
        for trial in reps:
            try:
                print(f"DEBUG: Calibrate using S{trial.subject_id}A{trial.action_id}T{trial.sequence_number}")
                # We'll do a partial approach: 
                #  1) load watch accel
                #  2) load watch gyro
                #  3) load skeleton
                #  4) align them
                #  5) extract orientation
                #  6) run calibrate_filter

                # (For brevity, let's just do a partial alignment. We'll skip phone here.)
                modkeys = trial.files.keys()
                accel_fp = trial.files['accelerometer']
                gyro_fp = trial.files['gyroscope']
                skel_fp = trial.files['skeleton']

                # load 
                accel_proc = Processor(file_path=accel_fp, mode=self.mode, max_length=self.max_length)
                gyro_proc  = Processor(file_path=gyro_fp, mode=self.mode, max_length=self.max_length)
                skel_proc  = Processor(file_path=skel_fp, mode=self.mode, max_length=self.max_length, is_skeleton=True)

                accel_data = accel_proc.load_file(is_skeleton=False)
                gyro_data  = gyro_proc.load_file(is_gyroscope=True)
                skel_data  = skel_proc.load_file(is_skeleton=True)

                if accel_data.size==0 or gyro_data.size==0 or skel_data.size==0:
                    print("DEBUG: Missing or empty data in calibration trial => skip")
                    continue

                # Interp gyro to accel
                from scipy.interpolate import interp1d
                if not np.array_equal(accel_data[:,0], gyro_data[:,0]):
                    ginterp = interp1d(gyro_data[:,0], gyro_data[:,1:], axis=0, fill_value='extrapolate', bounds_error=False)
                    aligned_gyro = ginterp(accel_data[:,0])
                    gyro_aligned = np.column_stack([accel_data[:,0], aligned_gyro])
                else:
                    gyro_aligned = gyro_data

                # robust align skeleton => method = self.align_method
                # (Simple approach)
                aligned_accel, aligned_skel, timestamps = robust_align_modalities(
                    accel_data, 
                    skel_data, 
                    accel_data[:,0],
                    method=self.align_method
                )
                if aligned_accel.shape[0]==0 or aligned_skel.shape[0]==0:
                    print("DEBUG: No overlap skeleton in calibration => skip")
                    continue

                # extract orientation from skeleton
                ref_orient = extract_orientation_from_skeleton(
                    aligned_skel[:,1:],  # skip time col
                    num_joints=32
                )

                # calibrate
                # calibrate_filter => user-labeled function that tries small param variations
                # returning best [process_noise, measurement_noise, gyro_bias_noise]
                filter_obj, param_arr = calibrate_filter(
                    accel= aligned_accel[:,1:],  # skip time col
                    gyro= None,  # or partial
                    skeleton= aligned_skel[:,1:],
                    filter_type=self.imu_fusion,
                    timestamps= aligned_accel[:,0]
                )
                # param_arr => e.g. [pnoise, mnoise, gbias]
                all_params.append(param_arr)

            except Exception as e:
                print(f"ERROR calibrating trial => {e}")
                traceback.print_exc()
                continue

        if all_params:
            avg_params = np.mean(all_params, axis=0)
            self.filter_params = {
                'process_noise': float(avg_params[0]),
                'measurement_noise': float(avg_params[1]),
                'gyro_bias_noise': float(avg_params[2])
            }
            self.calibrated = True
            print(f"DEBUG: Done calibrating => {self.filter_params}")
            self.save_calibration_parameters()
        else:
            print("DEBUG: All calibration attempts failed => keep defaults.")

    ############################################################################
    # The main "process_trial" merges advanced calibration but also fallback
    ############################################################################
    def process_trial(self, trial, subjects):
        if trial.subject_id not in subjects:
            return None

        # Must have watch accelerometer or we skip entirely (like approach B).
        if 'accelerometer' not in trial.files:
            print(f"DEBUG: Missing watch accel => skip S{trial.subject_id}A{trial.action_id}T{trial.sequence_number}")
            return None

        # Possibly load from cache
        cache_file = self.get_cache_filename(trial, subjects)
        if self.use_cache and os.path.exists(cache_file):
            try:
                loaded = np.load(cache_file, allow_pickle=True)
                return (loaded['processed_data'][()], loaded['labels'])
            except Exception as e:
                print(f"[WARN] Could not load cache {cache_file}: {e}")

        # Ok let's load watch accel, watch gyro if exist, phone accel/gyro if exist, skeleton if exist
        label = self._trial_label(trial)
        modkeys = trial.files.keys()

        watch_accel_fp = trial.files['accelerometer']  # must exist
        watch_gyro_fp  = trial.files.get('gyroscope','')  # might or might not exist
        phone_accel_fp = trial.files.get('phone_accelerometer','')  # partial fallback
        phone_gyro_fp  = trial.files.get('phone_gyroscope','')
        skeleton_fp    = trial.files.get('skeleton','')

        # 1) Load watch accel
        watch_acc_proc = Processor(
            file_path=watch_accel_fp, 
            mode=self.mode, 
            max_length=self.max_length,
            window_size_sec=self.window_size_sec,
            stride_sec=self.stride_sec
        )
        watch_acc_data = watch_acc_proc.load_file(is_skeleton=False)
        if watch_acc_data.size==0:
            print(f"DEBUG: Watch accel empty => skip trial")
            return None

        # 2) Load watch gyro if exist
        has_watch_gyro = os.path.exists(watch_gyro_fp)
        watch_gyro_data = np.zeros((0,4),dtype=np.float32)
        if has_watch_gyro:
            try:
                watch_gyro_proc = Processor(
                    file_path=watch_gyro_fp, 
                    mode=self.mode,
                    max_length=self.max_length,
                    window_size_sec=self.window_size_sec,
                    stride_sec=self.stride_sec
                )
                watch_gyro_data = watch_gyro_proc.load_file(is_gyroscope=True)
                if watch_gyro_data.size==0:
                    print(f"DEBUG: watch gyro missing or empty => fallback to watch-accel only")
                    has_watch_gyro=False
            except Exception as e:
                print(f"DEBUG: watch gyro error => {e}")
                has_watch_gyro=False

        # 3) Load phone data if exist
        has_phone = False
        phone_acc_data = np.zeros((0,4),dtype=np.float32)
        phone_gyro_data= np.zeros((0,4),dtype=np.float32)
        if phone_accel_fp and os.path.exists(phone_accel_fp):
            # We can do partial phone usage
            try:
                phone_acc_proc = Processor(
                    file_path=phone_accel_fp,
                    mode=self.mode,
                    max_length=self.max_length
                )
                phone_acc_data = phone_acc_proc.load_file(is_skeleton=False)
                if phone_acc_data.size>0:
                    has_phone = True
            except:
                pass
        if phone_gyro_fp and os.path.exists(phone_gyro_fp):
            try:
                phone_gyro_proc = Processor(
                    file_path=phone_gyro_fp,
                    mode=self.mode,
                    max_length=self.max_length
                )
                gdata = phone_gyro_proc.load_file(is_gyroscope=True)
                if gdata.size>0:
                    # If phone_acc_data is also >0, we have a phone
                    phone_gyro_data = gdata
                else:
                    has_phone = False
            except:
                has_phone=False
        if has_phone and (phone_acc_data.size==0 or phone_gyro_data.size==0):
            # We can't do advanced phone-based fusion if partial phone data is incomplete
            has_phone=False

        # 4) Load skeleton if exist
        has_skel = (skeleton_fp and os.path.exists(skeleton_fp))
        skel_data = np.zeros((0,97), dtype=np.float32)
        if has_skel:
            try:
                skel_proc= Processor(file_path=skeleton_fp, mode=self.mode, max_length=self.max_length)
                tmp_skel = skel_proc.load_file(is_skeleton=True)
                if tmp_skel.size>0:
                    skel_data = tmp_skel
                else:
                    if self.skel_error_strategy=='drop_trial':
                        return None
                    has_skel=False
            except:
                if self.skel_error_strategy=='drop_trial':
                    return None
                has_skel=False

        # => Now we do approach A style "IMU fusion with calibration" if watch+gyro + phone + skeleton are present 
        # => or watch+gyro + skeleton at least. But if phone is partial or missing => fallback approach B style
        # => If watch_gyro is missing => fallback watch-only

        # 5) Prepare to fuse
        # first let's define an "fused_imu" if we have watch_gyro or phone_??? 
        # We'll unify logic
        def do_imu_fusion(acc_data, gyr_data, filter_params):
            """
            A mini function that uses the chosen self.imu_fusion with the filter_params
            to produce fused data => shape(N, #features).
            For code brevity, we skip advanced partial merges or phone calibrations.
            """
            if self.imu_fusion=='standard':
                fil = StandardKalmanIMU(
                    process_noise=filter_params['process_noise'],
                    measurement_noise=filter_params['measurement_noise'],
                    gyro_bias_noise=filter_params['gyro_bias_noise']
                )
            elif self.imu_fusion=='ekf':
                fil = ExtendedKalmanIMU(
                    process_noise=filter_params['process_noise'],
                    measurement_noise=filter_params['measurement_noise'],
                    gyro_bias_noise=filter_params['gyro_bias_noise']
                )
            else:
                fil = UnscentedKalmanIMU(
                    process_noise=filter_params['process_noise'],
                    measurement_noise=filter_params['measurement_noise'],
                    gyro_bias_noise=filter_params['gyro_bias_noise']
                )
            # We interpret the first col as time 
            # The actual filter processes shape(N,3) for accel, shape(N,3) for gyro
            # We do a quick interpolation to align gyr => acc timestamps
            a_ts = acc_data[:,0]
            a_xyz= acc_data[:,1:]
            g_ts = gyr_data[:,0]
            g_xyz= gyr_data[:,1:]
            if not np.array_equal(a_ts, g_ts):
                from scipy.interpolate import interp1d
                ginterp = interp1d(g_ts, g_xyz, axis=0, fill_value='extrapolate', bounds_error=False)
                g_aligned = ginterp(a_ts)
            else:
                g_aligned = g_xyz
            # fuse
            fused_array = fil.process_sequence(a_xyz, g_aligned, timestamps=a_ts)
            return np.column_stack([a_ts, fused_array]) # shape(N, 1 + #fusedfeatures)

        fused_imu = None

        # If watch_gyro is present => we can do watch-based calibration. 
        # If phone is present => we can incorporate phone? (In principle we might unify watch+phone, but for brevity let's do watch-only or phone-only.)
        # We'll do advanced if phone is present, else fallback watch only. 
        # We'll do advanced approach A "calibrated" if has_gyro is True and (self.calibrated==True or we just use filter_params)

        # If user wants calibration but hasn't done it => 
        # we do it once. We'll do it outside in make_dataset, i.e. we do calibrate_filter_parameters in make_dataset, not here.

        # => Attempt fusion
        watch_fused = None
        phone_fused = None

        if has_watch_gyro:
            # advanced watch+gyro
            watch_fused = do_imu_fusion(watch_acc_data, watch_gyro_data, self.filter_params)
        if has_phone:
            # phone approach if we want. We'll produce phone_fused, but do we unify phone + watch?
            # For brevity, let's skip unifying phone+watch => partial code. 
            phone_fused = do_imu_fusion(phone_acc_data, phone_gyro_data, self.filter_params)

        # => If watch_fused is None => fallback approach B => watch accel only
        # => If watch_fused is not None => we store it as 'fused_imu'
        # => If phone_fused is also not None => we skip or combine? 
        #     Let's keep it simpler => if phone_fused => we choose phone if it's not missing windows at end
        #     We'll do dynamic fallback if phone is missing windows => watch. 
        # But user specifically said "Implement a hybrid approach using A unless there's missing windows => then B." 
        # We'll check #samples in phone_fused => if < threshold => fallback watch_fused => if that also fails => watch accel only.

        # For demonstration:
        final_imu_source='watch_only'
        final_imu_data = watch_acc_data  # default => watch accel only

        if watch_fused is not None and watch_fused.shape[0]>10:
            final_imu_source='watch_fused'
            final_imu_data= watch_fused

        if phone_fused is not None and phone_fused.shape[0]> 10:
            # check if it has missing windows at the end => 
            # We'll do a naive approach => if phone_fused shape< watch_fused shape => fallback
            if phone_fused.shape[0]>= (final_imu_data.shape[0] - 5):
                final_imu_source='phone_fused'
                final_imu_data= phone_fused
            else:
                print("DEBUG: phone_fused has missing windows => fallback watch_fused")

        # 6) Now we align skeleton if has_skel => approach A style robust alignment
        if has_skel and skeleton_fp and skel_data.shape[0]>0:
            try:
                aligned_imu, aligned_skel, new_ts = robust_align_modalities(
                    final_imu_data, # shape(N, 1 + ...)
                    skel_data,
                    final_imu_data[:,0],
                    method=self.align_method
                )
                if aligned_imu.shape[0]>0 and aligned_skel.shape[0]>0:
                    final_imu_data = aligned_imu
                    skel_data      = aligned_skel
                else:
                    if self.skel_error_strategy=='drop_trial':
                        print("DEBUG: alignment => no overlap => drop trial")
                        return None
                    has_skel=False
            except:
                if self.skel_error_strategy=='drop_trial':
                    print("DEBUG: skeleton alignment error => drop trial")
                    return None
                has_skel=False

        # 7) produce windows => min-len approach
        trial_processed = {}
        label_list = []

        def do_windowing(arr, is_skel=False):
            # we rely on Processor for window
            proc = Processor(
                file_path='',
                mode=self.mode,
                max_length=self.max_length,
                window_size_sec=self.window_size_sec,
                stride_sec=self.stride_sec
            )
            # if final_imu_source= 'watch_fused' => is_fused=True to fixed?
            is_fused= (final_imu_source.endswith('_fused') and (not is_skel))
            out_wins= proc.process(arr, is_fused=is_fused)
            return out_wins

        # watch accel or fused => "accelerometer" key
        # skeleton => "skeleton" key
        # we skip phone ?

        # If final_imu_source in ['watch_fused','phone_fused'] => store as 'fused_imu'
        # else => store as 'accelerometer'
        if final_imu_source.endswith('_fused'):
            wname= 'fused_imu'
        else:
            wname= 'accelerometer'

        # window the final_imu_data
        imu_wins= do_windowing(final_imu_data, is_skel=False)
        if len(imu_wins)==0:
            print("DEBUG: no windows => skip trial")
            return None
        trial_processed[wname]= imu_wins

        # skeleton
        if has_skel and skel_data.size>0:
            skel_wins= do_windowing(skel_data, is_skel=True)
            if len(skel_wins)==0:
                if self.skel_error_strategy=='drop_trial':
                    return None
                # else skip skeleton
            else:
                # match min-len
                min_len= min(len(imu_wins), len(skel_wins))
                trial_processed[wname] = imu_wins[:min_len]
                trial_processed['skeleton']= skel_wins[:min_len]
                label_list= [label]*min_len
        else:
            # no skeleton => just use the imu wins
            trial_processed[wname] = imu_wins
            label_list= [label]* len(imu_wins)

        # Done => store to cache if use_cache
        if self.use_cache:
            try:
                np.savez_compressed(cache_file, processed_data=trial_processed, labels=label_list)
            except Exception as e:
                print(f"[WARN] Cache save {cache_file} => {e}")

        return (trial_processed,label_list)

    ############################################################################
    # The main "make_dataset"
    ############################################################################
    def make_dataset(self, subjects, max_workers=12):
        """
        1) Possibly calibrate filter if do_calibration
        2) Multi-thread process each trial
        3) store self.processed_data
        """
        self.data.clear()
        self.processed_data.clear()
        self.processed_data['labels'] = []

        # If user wants calibration => do it if we have watch+gyro + skeleton for a few trials
        if self.do_calibration and self.imu_fusion in ['standard','ekf','ukf']:
            self.calibrate_filter_parameters(subjects)  # approach A style

        tasks = [(trial, subjects) for trial in self.dataset.matched_trials]
        print(f"INFO: Processing {len(tasks)} trials => fusion={self.imu_fusion}, fallback logic, calibration={self.calibrated}")

        start_t= time.time()
        with ProcessPoolExecutor(max_workers=max_workers) as exe:
            futs= [exe.submit(self.process_trial, t, s) for (t,s) in tasks]
            for fut in as_completed(futs):
                res= fut.result()
                if res is None:
                    continue
                trial_proc, lab_list= res
                # merge
                for k in trial_proc:
                    if k not in self.processed_data:
                        self.processed_data[k]= []
                    self.processed_data[k].extend(trial_proc[k])
                self.processed_data['labels'].extend(lab_list)

        end_t= time.time()
        print(f"INFO: Done building dataset in {end_t - start_t:.2f}s => total windows={len(self.processed_data['labels'])}")

        # e.g. if we want to unify e.g. 'fused_imu' => orientation => etc. 
        # approach A does that. We'll do something minimal. 
        # e.g. if fused_imu in data => replace 'accelerometer' ?

        # debug
        for kk in self.processed_data:
            if kk=='labels':
                print(f"DEBUG: {kk} => #items={len(self.processed_data[kk])}")
            else:
                print(f"DEBUG: {kk} => #windows={len(self.processed_data[kk])}")

        return self.processed_data

    def normalization(self):
        """
        If needed, do standard scaling of data, ignoring 'labels' & timestamps
        """
        # approach A style
        from sklearn.preprocessing import StandardScaler
        # skip if variable_time
        if self.mode=='variable_time':
            return self.processed_data

        for k in self.processed_data:
            if k=='labels':
                continue
            big= np.concatenate(self.processed_data[k], axis=0)
            shape_= big.shape
            feat_dim= shape_[-1]
            flat= big.reshape(-1, feat_dim)
            scal= StandardScaler().fit(flat)
            new_list=[]
            for arr in self.processed_data[k]:
                s= arr.shape
                f= arr.reshape(-1, feat_dim)
                f_= scal.transform(f)
                new_list.append(f_.reshape(s))
            self.processed_data[k] = new_list
        return self.processed_data

