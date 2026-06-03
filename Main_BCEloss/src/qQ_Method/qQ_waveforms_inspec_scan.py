import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import pickle
import tensorflow as tf
import sionna.phy as sn
from sionna.phy.channel.tr38901 import TDL
from src.qQ_Method.qQ_Model import qQ_MODEL
from config import SEED, CARRIER_FREQ
import os
import glob
from pathlib import Path
sn.config.seed = SEED


def prepare_model(model_class, weights_path=None, build_args=(1, 0.0), **kwargs):
    r'''
        Load weights of a model class, if exsits
    '''
    model = model_class(**kwargs)
    model(*build_args)  # call the model once to build so weights could be loaded
    if weights_path:
        with open(weights_path, 'rb') as f:
            weights = pickle.load(f)
            model.set_weights(weights)
    return model


def move_png_files(delay_spread_str):
    r'''
        Delete PNG files ending with '_min_DS', rename files ending with '_max_DS' (removing _max_DS),
        and move remaining PNG files starting with '_' to a subdirectory named after the delay_spread 
        inside the waveforms_ds_scan parent directory
    '''
    # Delete PNG files ending with '_min_DS'
    min_ds_files = glob.glob("*_min_DS.png")
    for min_file in min_ds_files:
        Path(min_file).unlink()
    
    # Rename PNG files ending with '_max_DS' by removing '_max_DS'
    max_ds_files = glob.glob("*_max_DS.png")
    for max_file in max_ds_files:
        new_name = max_file.replace("_max_DS.png", ".png")
        Path(max_file).rename(new_name)
    
    # Create parent directory
    parent_dir = Path("waveforms_ds_scan")
    parent_dir.mkdir(exist_ok=True)
    
    # Create subdirectory name based on delay_spread
    subdir_name = f"delay_spread_{delay_spread_str}"
    subdir_path = parent_dir / subdir_name
    
    # Create the subdirectory if it doesn't exist
    subdir_path.mkdir(exist_ok=True)
    
    # Find and move all PNG files starting with '_'
    png_files = glob.glob("_*.png")
    for png_file in png_files:
        src = Path(png_file)
        dst = subdir_path / png_file
        src.rename(dst)
        print(f"Moved {png_file} to {subdir_path}/")
    
    return subdir_path


if __name__ == "__main__":
    # Configuration for delay_spread loop
    start_delay_spread = 10e-9   # Start value in seconds
    end_delay_spread = 620e-9     # End value in seconds
    step_delay_spread = 10e-9    # Step size in seconds
    
    # Loop through delay_spread values
    current_delay_spread = start_delay_spread
    while current_delay_spread <= end_delay_spread:
        print(f"\n{'='*60}")
        print(f"Processing delay_spread: {current_delay_spread*1e9:.0f} ns")
        print(f"{'='*60}")
        
        qQ_model = prepare_model(qQ_MODEL, 'weights-qQ_Method')
        qQ_model._channel_model = TDL(model="A", delay_spread=current_delay_spread, 
                                       carrier_frequency=CARRIER_FREQ, 
                                       min_speed=0.0, max_speed=0.0)
        qQ_model.visulaize_progress = True
        qQ_model.training = True
        qQ_model(1, 40)
        
        # Move PNG files to subdirectory named after delay_spread
        delay_spread_str = f"{current_delay_spread*1e9:.0f}ns"
        move_png_files(delay_spread_str)
        
        # Increment delay_spread
        current_delay_spread += step_delay_spread
    
    print(f"\n{'='*60}")
    print("All processing completed!")
    print(f"{'='*60}")    