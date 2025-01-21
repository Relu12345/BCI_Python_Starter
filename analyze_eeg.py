import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_and_prepare_data(file_path):
    # Read the CSV file
    data = pd.read_csv(file_path)
    
    # Extract EEG channels (first 8 columns)
    eeg_data = data.iloc[:, 0:8].T
    
    # Create MNE info structure
    ch_names = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']
    ch_types = ['eeg'] * 8
    sfreq = 250  # Unicorn sampling frequency
    
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    
    # Create raw object
    raw = mne.io.RawArray(eeg_data, info)

    # Design filters
    #-------------------------------------------------------------------------------------
    # raw.filter - Bandpass filter that allows frequencies between 2 values (right now 1Hz and 30Hz) to pass through
    # Also used to keep relevant brain activity frequencies

    # notch_filter - Notch filter to remove a certain value's noise and power line interferences (our value is 50Hz)

    # scaling_factor (in next function) - Used to scale the data to a range of 100 microvolts
    raw.filter(l_freq=1, h_freq=30)
    raw.notch_filter(50)
    
    # Set montage (Unicorn uses 8 channels)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, match_case=False)
    
    return raw

def analyze_eeg(file_path):

    scaling_factor = 100e-6
    raw = load_and_prepare_data(file_path)
    
    # Plot raw data
    #-------------------------------------------------------------------------------------
    # Useful for signal preprocessing, artifact rejection, quality assessment etc.
    # With it, you can identify artifcats, eye blinks, muscle movements etc.
    raw.plot(title='Raw EEG Data', scalings=dict(eeg=scaling_factor), n_channels=8, show=True)
    
    # Create and plot PSD (fmax is used to limit the frequency range to our bandpass filter 30Hz)
    #-------------------------------------------------------------------------------------
    # Used to visualize the power spectral density (PSD) of the EEG data
    # Useful for anaylzing dominant frequencies, compare power between conditions.
    # Usually utilizied in quantifying brain rhythms and identifying state changes (e.g. rest vs activity)
    spectrum = raw.compute_psd(fmax=30)
    spectrum.plot(picks='eeg', average=True, dB=True)
    
    # Create epochs for better visualization
    events = mne.make_fixed_length_events(raw, duration=1.0)
    epochs = mne.Epochs(raw, events, tmin=0, tmax=1.0, baseline=None, preload=True)
    evoked = epochs.average()
    
    # Plot time-frequency representation
    #-------------------------------------------------------------------------------------
    # Delta (0-4 Hz): Sleep, deep relaxation
    # Theta (4-8 Hz): Memory, emotional processing
    # Alpha (8-12 Hz): Relaxed wakefulness
    # Beta (12-30 Hz): Active thinking, focus

    frequencies = np.arange(1, 50, 1)
    epochs.plot_psd_topomap(bands=[(0, 4, 'Delta'), (4, 8, 'Theta'), 
                                  (8, 12, 'Alpha'), (12, 30, 'Beta')])

    # Add 3D brain visualization
    mne.datasets.fetch_fsaverage(verbose=True)
    subjects_dir = mne.datasets.sample.data_path() / 'subjects'

    # Create source space
    src = mne.setup_source_space('fsaverage', spacing='oct6', 
                                subjects_dir=subjects_dir)

    # Compute forward solution
    conductivity = (0.3,)  # for single layer
    model = mne.make_bem_model('fsaverage', conductivity=conductivity,
                              subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(model)
    
    fwd = mne.make_forward_solution(evoked.info, trans='fsaverage',
                                  src=src, bem=bem)
    
    # Compute inverse solution
    inverse_operator = mne.minimum_norm.make_inverse_operator(
        evoked.info, fwd, noise_cov=None, fixed=True)
    
    stc = mne.minimum_norm.apply_inverse(evoked, inverse_operator)
    
    brain = mne.viz.Brain('fsaverage', subjects_dir=subjects_dir,
                         alpha=0.1,
                         hemi='both',
                         title='3D Brain Activity Visualization')
    
    brain.add_data(stc.data,
                  vertices=stc.vertices,
                  time_label='time',
                  colormap='RdBu_r',
                  smoothing_steps=7)

    brain.add_sensors(evoked.info, trans='fsaverage')
    
    plt.show()
    return brain

def main():
    print("EEG Analysis Tool")
    print("----------------")
    
    # Get file path from user
    file_path = input("Enter the path to your CSV file: ")
    
    try:
        brain_view = analyze_eeg(file_path)
        brain_view.show_view('lateral')
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
