#This module describes the preprocessing that is being done on to the datas,
#including, data loading, sliding window method, feature extraction calls, and One hot encoder.
import os
import features
import numpy as np
from sklearn.preprocessing import OneHotEncoder

#Global Variables
data_folder = "Data"
sampling_rate = 5
window_size = 2.56  #2.56 Seconds
overlapping_percentage = 0.5
table = {"windows.csv": 1, "pushback.csv": 2, "rocket.csv": 3, "elbowlock.csv": 4,
        "hair.csv": 5, "scarecrow.csv": 6, "zigzag.csv": 7, "shouldershrug.csv": 8,
        "left.csv": 9, "right.csv": 10, "idle.csv": 11, "logout.csv": 12}

#This method takes in the the data folder that the training csv data files is stored and 
#iterate through all these files, extracting the features out of them and append them to 
#the main list of features and labels.
def load():
    features = []
    labels = []
    for name_folder in os.listdir(data_folder):
        name_path = os.path.join(data_folder, name_folder)
        for csv_file in os.listdir(name_path):
            print("Processing:", name_folder, csv_file)
            csv_path = os.path.join(name_path, csv_file)
            f_temp, l_temp = extract_features(csv_path, csv_file)
            features.extend(f_temp)
            labels.extend(l_temp)
    return np.asarray(features), np.asarray(labels).reshape(-1,1)

#This method takes in the full path to a csv file and opens that file, iterate through the file 
#using sliding window method predefined, and extract the features out of the window slices by calling
#extract from features
def extract_features(csv_path, csv_file):
    data = []
    f_temp = []
    with open(csv_path, "r") as data_file: #open the files
        for row in data_file:
            strsplit = row.split(',')
            strsplit = list(map(float, strsplit))
            data.append(strsplit)

    no_of_data_per_window = int(sampling_rate * window_size)
    no_of_windows = len(data) // no_of_data_per_window * 2 - 1
    #count = 0
    for i in range(no_of_windows):
        window_slice_data = []
        for j in range(no_of_data_per_window):
            window_slice_data.append(data[i * no_of_data_per_window // 2 + j])
        #count += 1
        #print(count)
        window_slice_data = np.asarray(window_slice_data)[:, 0:3].tolist()
        f_temp.append(features.extract(window_slice_data))
    l_temp = np.full(len(f_temp), table[csv_file])
    return f_temp, l_temp

#This function takes in the labels and return them as a one hot encoded labels
def OHE_labels(labels):
    encoder = OneHotEncoder(sparse=False, dtype = float)
    OHE = encoder.fit_transform(labels)
    return OHE