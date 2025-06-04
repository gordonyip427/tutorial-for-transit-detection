"""
Kepler Light Curve Analysis for Exoplanet Detection
===================================================
This module contains utilities for loading, processing, and analyzing Kepler light curves
for exoplanet detection using CNN-based deep learning models.
"""

import numpy as np
import tensorflow as tf
import keras
import pandas as pd
import sklearn
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial

from keras import layers

#------------------------------------------------------------------------------
# DATA CONSTANTS AND FEATURE MAPPINGS
#------------------------------------------------------------------------------

# Label mapping for classification
# copied directly from https://github.com/google-research/exoplanet-ml/blob/master/exoplanet-ml/astronet/astro_model/configurations.py

label_map = {
    "PC": 1,    # Planet Candidate
    "AFP": 0,   # Astrophysical False Positive
    "NTP": 0,   # Non-Transiting Phenomenon
    "SCR1": 0,  # TCE from scrambled light curve with SCR1 order
    "INV": 0,   # TCE from inverted light curve
    "INJ1": 1,  # Injected Planet
    "INJ2": 0,  # Simulated eclipsing binary
}


#### WARNING: the following block of code is very specific to TFrecord dataset ####

# Create a description of each input feature. This is necessary to load in the dataset from TFrecord.
# Each key in this dictionary represents a feature in our dataset, with specifications for how to parse it.
# The features include both light curve data (global_view, local_view) and physical parameters.

feature_description = {
    # Transit depth (relative flux decrease during transit)
    'tce_depth': tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0,allow_missing=True),
    
    # Full light curve view (phase-folded)
    'global_view': tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0,allow_missing=True),
    
    # AstroNet predicted classification (planet/not-planet)
    'av_pred_class': tf.io.FixedLenSequenceFeature([], tf.string, default_value='',allow_missing=True),
    
    # Time of first transit (BJD - 2454833)
    'tce_time0bk': tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0.0,allow_missing=True),
    
    # Planet candidate number for this star
    'tce_plnt_num': tf.io.FixedLenSequenceFeature([], tf.int64, default_value=0,allow_missing=True),
    
    # Zoomed-in view around transit event
    'local_view': tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0.0,allow_missing=True),
    
    # Transit duration (hours)
    'tce_duration': tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0.0,allow_missing=True),
    
    # Training set designation (train/val/test)
    'av_training_set': tf.io.FixedLenSequenceFeature([], tf.string, default_value='',allow_missing=True),
    
    # Spline fitting breakpoint spacing
    'spline_bkspace': tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0.0,allow_missing=True),
    
    # Orbital period (days)
    'tce_period': tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0.0,allow_missing=True),
    
    # Model signal-to-noise ratio
    'tce_model_snr': tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0.0,allow_missing=True),
    
    # Kepler star ID
    'kepid': tf.io.FixedLenSequenceFeature([], tf.int64, default_value=0,allow_missing=True),
    
    # Impact parameter (transit geometry)
    'tce_impact': tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0.0,allow_missing=True),
    
    # Maximum Multi-Event Statistic value
    'tce_max_mult_ev': tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0.0,allow_missing=True),
    
    # Planetary radius (Earth radii)
    'tce_prad': tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0.0,allow_missing=True),
}

#------------------------------------------------------------------------------
# DATA LOADING AND PROCESSING
#------------------------------------------------------------------------------

def parse_function(example_proto, feature_description):
    """
    Parse a single TFRecord example using the feature description dictionary.
    
    Args:
        example_proto: A serialized tf.train.Example protocol buffer
        feature_description: Dictionary describing the features
    
    Returns:
        A dictionary of parsed features
    """
    return tf.io.parse_single_example(example_proto, feature_description)

def get_data(file_signature, feature_description):
    """
    Load and extract local view, global view, and label data from TFRecord files.
    
    Args:
        file_signature (str): Glob pattern to match TFRecord files (e.g., 'data/*.tfrecord')
        feature_description: Dictionary describing the TFRecord features
    
    Returns:
        tuple: (all_lv_data, all_gv_data, all_label)
            - all_lv_data: numpy array of shape (N, 201) containing local view data
            - all_gv_data: numpy array of shape (N, 2001) containing global view data
            - all_label: numpy array of shape (N, 1) containing classification labels
    """
    # Initialize empty arrays to store accumulated data
    # Local view: 201 points around transit
    all_lv_data = np.empty((0, 201))
    # Global view: 2001 points for full phase-folded light curve
    all_gv_data = np.empty((0, 2001))
    # Labels for classification
    all_label = np.empty((0, 1))
    
    # Find all TFRecord files matching the pattern
    files = glob.glob(file_signature)
    
    if not files:
        raise ValueError(f"No files found matching pattern: {file_signature}")
    
    for one_file in files:
        # Create TFRecord dataset from file
        dataset = tf.data.TFRecordDataset(one_file)
        
        # Count total number of examples in the file
        # This is necessary to batch all records at once
        N = sum(1 for _ in tf.data.TFRecordDataset(one_file))
        parser = partial(parse_function, feature_description=feature_description)
        
        # Parse the TFRecord using our custom parse function
        parsed_dataset = dataset.map(parser)
        
        # Extract all data as a single batch for efficiency
        # Note: This approach loads entire file into memory at once
        one_data = tf.data.Dataset.get_single_element(parsed_dataset.batch(N))
        
        # Extract specific features and convert to numpy arrays
        lv_data = one_data['local_view'].numpy()
        gv_data = one_data['global_view'].numpy()
        
        # Convert labels to UTF-8 encoding with fixed length
        # This ensures consistent string format across all files
        labels = one_data['av_training_set'].numpy().astype('U13')
        
        # Reshape labels to 2D array if necessary
        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)
        
        # Concatenate data from current file with accumulated data
        all_lv_data = np.concatenate([all_lv_data, lv_data], axis=0)
        all_gv_data = np.concatenate([all_gv_data, gv_data], axis=0)
        all_label = np.concatenate([all_label, labels], axis=0)
    
    print(f"Loaded {len(files)} files containing {all_lv_data.shape[0]} examples")
    
    return all_lv_data, all_gv_data, all_label

def to_numeric_encoding(all_label):
    """
    Convert string labels to numeric values using the label map.
    
    Args:
        all_label: Array of string labels
        
    Returns:
        numpy.ndarray: Array of numeric labels
    """
    df = pd.DataFrame(all_label)
    all_label = df[0].map(label_map).values
    return all_label

#------------------------------------------------------------------------------
# DATA VISUALIZATION
#------------------------------------------------------------------------------

def plot_local_and_global_view(global_view, local_view, ax, name):
    """
    Plot local and global views of a planetary candidate (PC) light curve.
    
    Args:
        global_view (array): Full phase-folded light curve data
        local_view (array): Zoomed-in view around transit event
        ax (array): Array of two matplotlib axes objects for plotting
        name (str): Identifier for the planetary candidate
    """
    # Plot global view - full phase-folded light curve
    ax[0].plot(global_view)
    ax[0].set_title(f'{name} / Global View')
    ax[0].set_xlabel('Timesteps')
    ax[0].set_ylabel('Normalized Flux')
    
    # Plot local view - zoomed transit region
    ax[1].plot(local_view)
    ax[1].set_title(f'{name} / Local View (+Smoothing)')
    ax[1].set_xlabel('Timesteps')
    ax[1].set_ylabel('Normalized Flux')

def visualize_planetary_candidates(training_gview, training_lview, PC_index,name= None,save=False):
    """
    Create a visualization grid showing multiple planetary candidates.
    
    Args:
        training_gview (array): Global view data for all training examples
        training_lview (array): Local view data for all training examples
        PC_index (list): Indices of planetary candidates to visualize
    """
    # Create a 3x2 grid of subplots
    fig, ax = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(10, 6),
        sharey=True  # Share y-axis for better comparison
    )
    
    # Plot each planetary candidate
    for i in range(3):
        current_pc_index = PC_index[i]
        plot_local_and_global_view(
            global_view=training_gview[current_pc_index],
            local_view=training_lview[current_pc_index],
            ax=ax[i, :],
            name=f'ID:{current_pc_index}'
        )
    if name is not None:
        fig.suptitle(f'{name}')
    # Adjust layout and display
    plt.tight_layout()
    if save:
        if name is None:
            raise ValueError('please provide a name')
        plt.savefig(f'{name}.pdf')
    plt.show()

#------------------------------------------------------------------------------
# MODEL BUILDING BLOCKS
#------------------------------------------------------------------------------

def CNN_block(input_feature, filters, act='relu'):
    """
    Create a CNN block with a convolution layer followed by max pooling.
    
    Args:
        input_feature: Input tensor to the CNN block
        filters (int): Number of filters in the convolution layer
        act (str): Activation function to use (default: 'relu')
    
    Returns:
        tensorflow.Tensor: Output of the max pooling layer
    """
    # Convolution layer with kernel size 3
    cnn_1 = layers.Conv1D(filters, kernel_size=3, activation=act)(input_feature)
    
    # Max pooling for downsampling and feature reduction
    maxpool_1 = layers.MaxPool1D()(cnn_1)
    
    return maxpool_1

#------------------------------------------------------------------------------
# MODEL ARCHITECTURES
#------------------------------------------------------------------------------

def build_full_model(lview_shape, gview_shape, act_func):
    """
    Build a dual-input CNN model that processes both local and global views.
    
    Args:
        lview_shape (int): Shape of local view input
        gview_shape (int): Shape of global view input
        act_func (str): Activation function to use
        
    Returns:
        keras.Model: Compiled model with dual inputs
    """
    # Local View Branch
    input_local_view = keras.Input(shape=(lview_shape,))
    input_local_view_r = keras.layers.Reshape((-1, 1))(input_local_view)
    
    block_lv_1 = CNN_block(input_local_view_r, 16, act_func)
    block_lv_2 = CNN_block(block_lv_1, 32, act_func)
    flatten_lv = layers.Flatten()(block_lv_2)
    
    # Global View Branch
    input_global_view = keras.Input(shape=(gview_shape,))
    input_global_view_r = keras.layers.Reshape((-1, 1))(input_global_view)
    
    block_gv_1 = CNN_block(input_global_view_r, 16, act_func)
    block_gv_2 = CNN_block(block_gv_1, 32, act_func)
    block_gv_3 = CNN_block(block_gv_2, 32, act_func)
    block_gv_4 = CNN_block(block_gv_3, 32, act_func)
    block_gv_5 = CNN_block(block_gv_4, 32, act_func)
    flatten_gv = layers.Flatten()(block_gv_5)
    
    # Merge branches and add classification head
    concat_layer = layers.Concatenate()([flatten_lv, flatten_gv])
    dense_1 = layers.Dense(64, activation=act_func)(concat_layer)
    dense_2 = layers.Dense(64, activation=act_func)(dense_1)
    output = layers.Dense(1, activation="sigmoid")(dense_2)
    
    return keras.Model(
        inputs=[input_local_view, input_global_view],
        outputs=output,
        name="planet_classifier"
    )

def build_lv_model(lview_shape, act_func):
    """
    Build a CNN model that processes only local view data.
    
    Args:
        lview_shape (int): Shape of local view input
        act_func (str): Activation function to use
        
    Returns:
        keras.Model: Compiled model with local view input
    """
    input_local_view = keras.Input(shape=(lview_shape,))
    input_local_view_r = keras.layers.Reshape((-1, 1))(input_local_view)
    
    block_lv_1 = CNN_block(input_local_view_r, 16, act_func)
    block_lv_2 = CNN_block(block_lv_1, 32, act_func)
    flatten_lv = layers.Flatten()(block_lv_2)
    
    dense_1 = layers.Dense(32, activation=act_func)(flatten_lv)
    dense_2 = layers.Dense(32, activation=act_func)(dense_1)
    output = layers.Dense(1, activation="sigmoid")(dense_2)
    
    return keras.Model(
        inputs=input_local_view,
        outputs=output, 
        name="planet_classifier_lv"
    )

def build_gv_model(gview_shape, act_func):
    """
    Build a CNN model that processes only global view data.
    
    Args:
        gview_shape (int): Shape of global view input
        act_func (str): Activation function to use
        
    Returns:
        keras.Model: Compiled model with global view input
    """
    input_global_view = keras.Input(shape=(gview_shape,))
    input_global_view_r = keras.layers.Reshape((-1, 1))(input_global_view)
    
    block_gv_1 = CNN_block(input_global_view_r, 16, act_func)
    block_gv_2 = CNN_block(block_gv_1, 32, act_func)
    block_gv_3 = CNN_block(block_gv_2, 32, act_func)
    block_gv_4 = CNN_block(block_gv_3, 32, act_func)
    block_gv_5 = CNN_block(block_gv_4, 32, act_func)
    flatten_gv = layers.Flatten()(block_gv_5)
    
    dense_1 = layers.Dense(64, activation=act_func)(flatten_gv)
    dense_2 = layers.Dense(64, activation=act_func)(dense_1)
    output = layers.Dense(1, activation="sigmoid")(dense_2)
    
    return keras.Model(
        inputs=input_global_view,
        outputs=output, 
        name="planet_classifier_gv"
    )

