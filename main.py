#This section describes the main algorithm for training the data
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import preprocessing
import numpy as np
import model

#preprocessing call, after this step, features and labels will be extracted out of the raw data
features, labels = preprocessing.load()
print("Feature shape:", np.asarray(features).shape)
print("Label shape:", np.asarray(labels).shape)

# Shuffle up the different moves
#features, labels = shuffle(features, labels)

#Train-Test Split with 7:3 Ratio/ Cross validation if there is not enough data
#features, labels = shuffle(features, labels)
# X_train, X_val, X_test = features[:int(len(features)*0.5), :], features[int(len(features)*0.5):int(len(features)*0.75), :], features[int(len(features)*0.75):, :]
# y_train, y_val, y_test = labels[:int(len(labels)*0.5), :], labels[int(len(labels)*0.5):int(len(labels)*0.75), :], labels[int(len(labels)*0.75):, :]
np.random.seed(1)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=1)

#One Hot Encoding labels
y_train_OHE = preprocessing.OHE_labels(y_train)
y_val_OHE = preprocessing.OHE_labels(y_val)
y_test_OHE = preprocessing.OHE_labels(y_test)

#print(X_train, X_val, X_test, y_train, y_val, y_test)

#Creating and Fitting the Model
#FFNN
es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=25)
model_ffnn = model.FFNN(input_dimension=X_train.shape[1], output_dimension=y_train_OHE.shape[1])
model_ffnn.fit(X_train, y_train_OHE, validation_data=(X_val, y_val_OHE), epochs=2000, batch_size=32, shuffle=True, callbacks=[es])
#RFC

#model.RFC(X_train, y_train)

#Result Evaluation