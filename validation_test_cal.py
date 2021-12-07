#@title Run on TensorFlow 2.x
%tensorflow_version 2.x

#@title Import relevant modules
import pandas as pd
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

# The following lines adjust the granularity of reporting. 
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

# Import the dataset.
train_df = pd.read_csv(filepath_or_buffer="housing_train_data.csv")

# Scale the label.
train_df["median_house_value"] /= 1000.0

test_df = pd.read_csv(filepath_or_buffer="housing_test_data.csv")
