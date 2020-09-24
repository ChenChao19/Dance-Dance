from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

from matplotlib import pyplot as plt

#Keras feed forward neural network
def FFNN(input_dimension, output_dimension):
    opt = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model = Sequential()
    model.add(Dense(256, input_dim=input_dimension, activation='relu', kernel_initializer='random_uniform', bias_initializer=initializers.Constant(0.1)))
    model.add(Dense(256, activation='relu', kernel_initializer='random_uniform', bias_initializer=initializers.Constant(0.1)))
    model.add(Dense(128, activation='relu', kernel_initializer='random_uniform', bias_initializer=initializers.Constant(0.1)))
    model.add(Dense(output_dimension, activation='sigmoid', kernel_initializer='random_uniform', bias_initializer='zeros'))
    model.compile(optimizer = opt, loss='categorical_crossentropy', metrics=['accuracy'])
    #model.summary()
    return model

#Scikit learn Random Forest Classifier
def RFC(X_train, y_train):
    y_train = y_train.reshape(len(y_train))
    print("Starting training")
    rfc = RandomForestClassifier(random_state=101)
    rfecv = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(10), scoring='accuracy')
    selector = rfecv.fit(X_train, y_train)
    print('Optimal number of features: {}'.format(rfecv.n_features_))
    plt.figure(figsize=(16, 9))
    plt.title('Recursive Feature Elimination with Cross-Validation', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Number of features selected', fontsize=14, labelpad=20)
    plt.ylabel('% Correct Classification', fontsize=14, labelpad=20)
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, color='#303F9F', linewidth=3)
    plt.show()
    print(selector.support_)
    print(selector.ranking_)