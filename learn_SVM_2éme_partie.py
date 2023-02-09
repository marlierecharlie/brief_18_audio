from matplotlib import pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn import svm
import pickle
from joblib import dump
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from features_functions import compute_features
import wave
import os
import librosa


# Initialize empty arrays to store features and labels
learning_Features = np.empty((0, 71))
learning_Labels = np.array([])

# LOOP OVER THE SIGNALS


path = "data" # chemin vers le dossier contenant les fichiers .wav

for filename in os.listdir(path):
    if filename.endswith(".wav"):
        filepath = os.path.join(path, filename)
        # utilisez votre fonction de lecture de fichier audio ici pour lire le fichier et le convertir en tableau numpy
        data, fs = librosa.load(filepath)
        # utilisez data pour ce que vous voulez faire ensuite
        input = data.astype(np.float64)
    input_sig = np.nan_to_num(input)
    

    # Compute the signal in three domains
    sig_sq = input_sig**2
    sig_t = input_sig / np.sqrt(sig_sq.sum())
    sig_f = np.absolute(np.fft.fft(sig_t))
    sig_c = np.absolute(np.fft.fft(sig_f))
    # plt.figure()
    # plt.plot(sig_t)
    # plt.show()
    #print(fs)

    # Compute the features and store them
    N_feat, features_list = compute_features(sig_t, sig_f[:sig_t.shape[0]//2], sig_c[:sig_t.shape[0]//2],fs)
    features_vector = np.array(features_list)[np.newaxis,:]
    learning_Features = np.vstack((learning_Features, features_vector))
    #print(f"{filename}--------------------------------------------------------\n",learning_Features)

    # Store the labels
    if filename.startswith("noise"): 
        label = "noise"
    elif filename.startswith("car"): 
        label = "car"
    elif filename.startswith("truck"): 
        label = "truck"
    learning_Labels= np.append(learning_Labels, label)
#print(learning_Labels)

X_train, X_test, y_train, y_test = train_test_split(learning_Features, learning_Labels, test_size=0.3, random_state=10)

# Encode the class names
labelEncoder = preprocessing.LabelEncoder().fit(y_train)
learningLabelsStd = labelEncoder.transform(y_train)
testLabelsStd = labelEncoder.transform(y_test)

# Learn the model
model = svm.SVC(C=10, kernel='linear', class_weight=None, probability=False)
scaler = preprocessing.StandardScaler(with_mean=True).fit(X_train)
learningFeatures_scaled = scaler.transform(X_train)
#print(learningFeatures_scaled)

model.fit(learningFeatures_scaled, learningLabelsStd)


#Test the model
testFeatures_scaled = scaler.transform(X_test)

print("------------------------------------")
print("Jeux de validation: ", y_test)
print("Prédiction : ", model.predict(testFeatures_scaled))
print("Score :" , model.score(testFeatures_scaled, testLabelsStd))
print("------------------------------------")

# Export the scaler and model on disk
dump(scaler, "SCALER")
dump(model, "SVM_MODEL")
# Matrix confusion
plot_confusion_matrix(model, testFeatures_scaled, testLabelsStd) 
plt.show()
