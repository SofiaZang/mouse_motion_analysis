import cv2
import tifffile as tifffile
import numpy as np
import os
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import label, binary_closing

def smooth_with_gaussian(var, sigma=None):
    smoothed_var = gaussian_filter1d(var, sigma=sigma)

    return smoothed_var

def compute_motion_energy(movie_path=None, xrange=None, yrange=None, save_path=None):
    """
    Compute motion energy from a multi-frame TIFF movie of mouse movement.

    Parameters:
    - movie_path: Path to the multi-frame TIFF file (None, unless ititial run)
    - xrange: Range of x-values to crop the image (optional)
    - yrange: Range of y-values to crop the image (optional)
    - save_path: Path to save the motion energy result (optional)
    """
    
    #check if motion energy has already been computed, if that's the case load it

    motion_energy_path = os.path.join(save_dir,"motion_energy.npy")
    if os.path.exists(motion_energy_path):

        print(f'Motion energy already computed. Loading from {motion_energy_path}')
        motion_energy = np.load(motion_energy_path)
        return motion_energy

    if not movie_path:

        raise ValueError('Please provide the tiff for the initial run') 
    
    # Load the TIFF movie (multi-frame TIFF)
    movie = tifffile.imread(movie_path)
    num_frames, height, width = movie.shape

    print(f'Loaded movie with {num_frames} frames, height={height}, width={width}')

    # Initialize motion energy array
    motion_energy = np.zeros(num_frames)
    img_prev = movie[0]

    # Iterate over the frames and compute motion energy
    for i in range(1, num_frames):
        img = movie[i]

        # Compute motion energy as squared differences between consecutive frames
        diff = img - img_prev
        squared_diff = diff ** 2
        motion_energy[i] = np.sum(squared_diff)

        # Update img_prev for the next iteration
        img_prev = img

        # Print progress every 1000 frames
        if i % 1000 == 0:
            print(f'Done computing for {i}/{num_frames} frames')
    
    # Normalize motion energy
    motion_energy = motion_energy[1:]  # Skip the first frame (no previous frame to compare)
    motion_energy /= np.max(motion_energy)
    
    return motion_energy

def compute_thresholds_for_bin_state_detection(motion_signal, save_dir=None, plot=True):
    '''
    Compute statistical thresholds to use for threshold-based state detection (active/awake - rest)
    Here we use: Otsu (prefer if binary distribution), Li (mutliple peak distribution), or mean+sd (gaussian distribution)
    '''
    
    # mean+sd threshold 
    motion_mean = np.mean(motion_signal)
    motion_sd = np.std(motion_signal)

    threshold_motion_mean_sd = motion_mean + motion_sd

    threshold_motion_li = filters.threshold_li(motion_signal)

    threshold_motion_otsu = filters.threshold_otsu(motion_signal)

    if plot:
        plt.figure(figsize=(5, 5), dpi=300)
        plt.title('stat thresholds on mot_en')
        plt.hist(motion_signal, bins=70, alpha=0.9)

        # mark threshold lines
        plt.axvline(x=threshold_motion_mean_sd, color='red', label='mean + sd', linestyle='--')
        plt.axvline(x=threshold_motion_otsu, color='salmon', label='Otsu', linestyle='--')
        plt.axvline(x=threshold_motion_li, color='darkred', label='Li', linestyle='--')

        plt.legend()
        plt.savefig(save_dir + 'binary_state_thresholds.png')
        plt.show()

    return threshold_motion_mean_sd, threshold_motion_li, threshold_motion_otsu
    
def binarise_motion(motion_signal, binary_threshold, min_duration):
    '''
    Binarises motion signal (eg mot_en) into 0s (rest) and 1s (active)
    input params:
    motion_signal: motion energy or other
    binary_threshold: sts threshold (li, otsu or mean_sd for state detection)
    min_di=uration: min_duration threshold to be detected as awake/active motion
    
    returns:
    bin_motion_signal: bin array of 0s and 1s 
    inds_active_state: indices of frames whens state is active
    inds_rest_state: indices of frames when state is rest/inactive 

    '''
    
    bin_motion_signal = np.zeros(len(motion_signal), dtype=int)

    # get boolean array 
    
    all_active_motions_detected = motion_signal > binary_threshold 

    # labels continuous segments that pass the threshold 

    labeled_array, n_features = label(all_active_motions_detected) #how many active motions where found 
    
    for i in range(1, n_features+1):
        segment = np.where(labeled_array == i)[0]
        if len(segment) > min_duration:
            bin_motion_signal[segment] = 1

    inds_active_state = np.where(bin_motion_signal ==1)
    inds_rest_state = np.where(bin_motion_signal ==0)
    
    return bin_motion_signal, inds_active_state, inds_rest_state

def classify_active_motion_segments(bin_motion_signal, motion_signal, short_threshold, long_threshold):
    """
    Classify active/awake motion segments into short (1–3 sec), long (>3 sec), and too short (<1 sec, those are exluded in awake motion detection and set to 0).

    Parameters:
    - bin_motion_signal: 1D array of HMM or thresholded motion states (1 for active, 0 for inactive)
    - motion_signal: 1D array of motion energy values
    - short_threshold: in frames, minimum duration for short active motions (e.g., 3 or 1s at 3Hz)
    - long_threshold: in frames, minimum duration for long active motions (e.g., 9 or 3s at 3Hz)

    Returns:
    - bin_short_active_motion: binary array marking short motions (1-3s)
    - bin_long_active_motion: binary array marking long motions (>3s)
    - bin_too_short_active_motion: binary array marking short blips (<1s)
    - and correspondings inds of long, short or too short (excluded) motions
    """
    labeled_array, num_features = label(bin_motion_signal == 1)

    bin_short_active_motion = np.zeros(len(motion_signal), dtype=int)
    bin_long_active_motion = np.zeros(len(motion_signal), dtype=int)
    bin_too_short_active_motion = np.zeros(len(motion_signal), dtype=int)

    for i in range(1, num_features + 1):
        segment = np.where(labeled_array == i)[0]

        if len(segment) > long_threshold:
            bin_long_active_motion[segment] = 1
        elif len(segment) > short_threshold:
            bin_short_active_motion[segment] = 1
        elif len(segment) > 0: # but shorter than short threshold 
            bin_too_short_active_motion[segment] = 0 # mark as inactive 

        long_active_motion_inds = np.where(bin_long_active_motion==1)[0]
        short_active_motion_inds = np.where(bin_short_active_motion==1)[0]
        too_short_active_motion_inds = np.where(bin_too_short_active_motion==1)[0]

            
    return bin_short_active_motion, short_active_motion_inds, bin_long_active_motion, long_active_motion_inds, bin_too_short_active_motion, too_short_active_motion_inds


'''
Define hyperparameters for twitch detection. Added a minimum distance (in frames) required 
for the twitch to be detected (this is to avoid inter-active-sleep periods artifacts, if the twitch happens too close 
too an active motion (precedes or follows) exclude it, becs we cannot be sure the mouse was asleep already or in between.
We only keep best, robust twitches 
'''
def filter_twitches_by_awake_proximity(inds_twitches, inds_active_state, min_distance=None):
    filtered_twitches = []

    for idx_twitch in inds_twitches:
        distance = np.abs(inds_active_state - idx_twitch)
        if np.min(distance) > min_distance:
            filtered_twitches.append(idx_twitch)

    filtered_twitches = np.array(filtered_twitches)

    return filtered_twitches

def filter_twitches_only_post_active_mot(inds_twitches, inds_active_state, min_distance=None):
    filtered_twitches = []

    for idx_twitch in inds_twitches:
        distances = inds_active_state - idx_twitch  # Compute relative distances
        closest_active_motion = np.min(distances[distances >= 0], initial=np.inf)  # Only consider future active motions

        # Exclude only if the twitch is after and within min_distance
        if closest_active_motion > min_distance:
            filtered_twitches.append(idx_twitch)

    return np.array(filtered_twitches)

def remove_twitch_bursts(twitch_binary, framerate, min_interval=1):
    """
    Removes all twitches that occur within `min_interval` seconds of another twitch.
    Both twitches in a burst/close proximimity are excluded.
    
    Parameters:
        twitch_binary (np.ndarray): Binary vector of twitch detections (1 = twitch, 0 = no twitch)
        fs (float): framerate (Hz)
        min_interval (float): minimum interval required between twitches for those to be detected (in sec)
    
    Returns:
        np.ndarray: filtered binary twitch array 
    """
    twitch_indices = np.where(twitch_binary == 1)[0]
    to_remove = set()
    
    # Compare each twitch to the next one
    for i in range(len(twitch_indices) - 1):
        curr_idx = twitch_indices[i]
        next_idx = twitch_indices[i + 1]
        interval = (next_idx - curr_idx) / framerate # in sec 
        
        if interval < min_interval:
            to_remove.add(curr_idx)
            to_remove.add(next_idx)
    
    filtered_twitches = np.copy(twitch_binary)
    filtered_twitches[list(to_remove)] = 0
    return filtered_twitches 

def filter_bursty_twitches(inds_twitches, min_frame_gap=3):
    """
    Keeps only the first twitch in bursts occurring within `min_frame_gap` frames.
    
    Parameters:
        inds_twitches (list of list of int): Each inner list contains frame indices for a twitch.
        min_frame_gap (int): Minimum allowed gap between two twitch events (in frames).
        
    Returns:
        list of list of int: Filtered list of twitch events (same structure).
    """
    if not inds_twitches:
        return []

    filtered = [inds_twitches[0]]
    last_kept = inds_twitches[0][0]  # take the first index of the first twitch

    for twitch in inds_twitches[1:]:
        current_onset = twitch[0]  # first index of current twitch
        if current_onset - last_kept >= min_frame_gap:
            filtered.append(twitch)
            last_kept = current_onset

    return filtered

def binarise_twitch(motion_energy, twitch_segments):

    # Initialize bin_twitch   
    bin_twitch = np.zeros(len(motion_energy))

    # flatten the nested list
    flat_inds = [idx for segment in twitch_segments for idx in segment] # filtered corrected twitch segments 

    print("Flattened Indices:", flat_inds)

    # Assign 1 to twitch inds
    for idx in flat_inds:
        if idx < len(motion_energy):  # Ensure index is within bounds
            bin_twitch[idx] = 1
            print(f"Assigning 1 at column {idx}")
        else:
            print(f"Skipping out-of-bounds index: {idx}")
    return bin_twitch


# Define a helper function to get discrete active state segments
''' modified to handle empty arrays
'''
def get_active_segments(active_indices):
    segments = []
    if active_indices.size == 0:
        return segments  # Return an empty list if there are no active indices
    start = active_indices[0]
    for i in range(1, len(active_indices)):
        # Check if the current index is not consecutive with the previous
        if active_indices[i] != active_indices[i - 1] + 1:
            end = active_indices[i - 1]
            segments.append((start, end))
            start = active_indices[i]
    segments.append((start, active_indices[-1]))
    return segments

def find_sequential_groups(arr):
    groups = []
    current_group = [arr[0]]
    
    for i in range(1, len(arr)):
        if arr[i] == arr[i-1] + 1:
            current_group.append(arr[i])
        else:
            groups.append(current_group)
            current_group = [arr[i]]
    groups.append(current_group)
    
    return groups

def get_onsets(bin_motion):
    onsets = np.where((bin_motion[1:] == 1) & (bin_motion[:-1] == 0))[0] + 1 #[0] specific to stcuture of array
    return onsets

def get_offsets(bin_motion):
    offsets = np.where((bin_motion[1:] == 0) & (bin_motion[:-1] == 1))[0]  #[0] here 0->1 we want the 0 as offset so not +1

# filter twitches based on duration 
def filter_segments_by_duration(segments, duration_threshold):
# Filter out groups based on the duration threshold
    return [twitches for twitches in segments if len(twitches) <= duration_threshold]


'''
Input a motion or batches of motiona nd output the average length of the motion in seconds 

'''

def get_length_of_motion(concatenated_motion, frame_rate): #can even be a xhole motion or concatenated motion segments
    if len(concatenated_motion) == 1: 
        mean_length_motion_sec = len(concatenated_motion[0]) / frame_rate
    
    mean_length_motion = np.mean([len(segment) for segment in concatenated_motion])

    mean_length_motion_sec = mean_length_motion / frame_rate

    return mean_length_motion_sec

# Define a helper function to get discrete active state segments
def get_active_segments(active_indices):
    segments = []
    start = active_indices[0]
    for i in range(1, len(active_indices)):
        # Check if the current index is not consecutive with the previous
        if active_indices[i] != active_indices[i - 1] + 1:
            segments.append((start, active_indices[i - 1]))
            start = active_indices[i]
    segments.append((start, active_indices[-1]))  # Add the last segment
    return segments

# compute correlation (coupling) between behaviour and pcs (smoothing a bit before computing corr)

def compute_corrs(behaviour, pcs):
    beh_pcs_coupling = []
    for i in range(pcs.shape[0]): #no. of pcs computed, time 
        corr = np.corrcoef(gaussian_filter1d(pcs[i,:], sigma=3), gaussian_filter1d(behaviour, sigma=3))[0,1]
        beh_pcs_coupling.append(corr)
    return beh_pcs_coupling

def get_onsets(bin_motion):
    onsets = np.where((bin_motion[1:] == 1) & (bin_motion[:-1] == 0))[0] + 1 #[0] specific to stcuture of array
    if bin_motion[0] == 1:  
        onsets = np.insert(onsets, 0, 0) # solves issue with twitch detected at first frames  
    return onsets
    
def get_offsets(bin_motion):
    offsets = np.where((bin_motion[1:] == 0) & (bin_motion[:-1] == 1))[0] + 1 #[0] specific to stcuture of array
    return offsets  