import numpy as np
import csv
import time
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


def classifer_metrics(y_true, y_pred):
    classifer_accuracy = accuracy_score(y_true, y_pred)*100
    classifier_f1_score = f1_score(y_true, y_pred, average='macro')*100
    return classifer_accuracy, classifier_f1_score

# This line reads the dataset names instead of hardcoding them.
datasets = os.listdir('./datasets')
iterations = 3

print(f"------------------------- SVM -----------------------------")


#SVM
training_time_SVM = {k:[] for k in datasets}
training_accuracy_SVM = {k:[] for k in datasets}
training_f1_score_SVM = {k:[] for k in datasets}
testing_time_SVM = {k:[] for k in datasets}
testing_accuracy_SVM = {k:[] for k in datasets}
testing_f1_score_SVM = {k:[] for k in datasets}


for dataset in datasets:
    print(f"------------- {dataset[:-4]} Execution -------------")
    with open(f'./datasets/{dataset}','r') as csv_file:
        csv_reader = csv.reader(csv_file)
        data = list(csv_reader)

    data = np.array(data)
    data = data.astype(np.float)
        
    num_observations = data.shape[0]
    num_features = data.shape[1] - 1

    print(f"Dataset Size: {data.shape}\n")

    for i in range(iterations):
        # Shuffle the data
        shuffle_idx = np.random.permutation(num_observations)
        shuffled_data = data[shuffle_idx,:]

        X = shuffled_data[:,:-1]
        y = shuffled_data[:,-1]

        le = LabelEncoder()
        y = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        
        classifier = SVC(kernel = 'poly', random_state = 0)

        training_time_start = time.time()
        
        classifier.fit(X_train, y_train)
        y_train_pred = classifier.predict(X_train)
        classifir_accuracy, classifier_f1_score = classifer_metrics(y_train, y_train_pred)
        training_accuracy_SVM[dataset].append(classifir_accuracy)
        training_f1_score_SVM[dataset].append(classifier_f1_score)

        training_time_end = time.time()
        training_time_SVM[dataset].append(training_time_end - training_time_start)

        testing_time_start = time.time()
        
        y_test_pred = classifier.predict(X_test)
        classifir_accuracy, classifier_f1_score = classifer_metrics(y_test, y_test_pred)
        testing_accuracy_SVM[dataset].append(classifir_accuracy)
        testing_f1_score_SVM[dataset].append(classifier_f1_score)
        
        testing_time_end = time.time()
        testing_time_SVM[dataset].append(testing_time_end - testing_time_start)

    print(f"\nAverage Training Time: --- {np.mean(training_time_SVM[dataset]):.4f} seconds ---")
    print(f"Average Training Accuracy: --- {np.mean(training_accuracy_SVM[dataset]):.2f}%")
    print(f"Average Training F1 Score: --- {np.mean(training_f1_score_SVM[dataset]):.2f}%")
    print(f"Average Testing Time: --- {np.mean(testing_time_SVM[dataset]):.4f} seconds ---")
    print(f"Average Testing Accuracy: --- {np.mean(testing_accuracy_SVM[dataset]):.2f}%")
    print(f"Average Testing F1 Score: --- {np.mean(testing_f1_score_SVM[dataset]):.2f}%\n")


print(f"------------------------- DNN -----------------------------")

    #DNN
training_time_DNN = {k:[] for k in datasets}
training_accuracy_DNN = {k:[] for k in datasets}
training_f1_score_DNN = {k:[] for k in datasets}
testing_time_DNN = {k:[] for k in datasets}
testing_accuracy_DNN = {k:[] for k in datasets}
testing_f1_score_DNN = {k:[] for k in datasets}

for dataset in datasets:
    print(f"------------- {dataset[:-4]} Execution -------------")
    with open(f'./datasets/{dataset}','r') as csv_file:
        csv_reader = csv.reader(csv_file)
        data = list(csv_reader)

    data = np.array(data)
    data = data.astype(np.float)
        
    num_observations = data.shape[0]
    num_features = data.shape[1] - 1

    print(f"Dataset Size: {data.shape}\n")
    for i in range(iterations):
        # Shuffle the data
        shuffle_idx = np.random.permutation(num_observations)
        shuffled_data = data[shuffle_idx,:]

        X = shuffled_data[:,:-1]
        Y = shuffled_data[:,-1]

        y_ = Y.reshape(-1, 1) # Convert data to a single column

        # normlize power column in dataset
        X[:,1] = (X[:,1] - X[:,1].mean())/(X[:,1].max()-X[:,1].min())

        # One Hot encode the class labels
        encoder = OneHotEncoder(sparse=False)
        Y = encoder.fit_transform(y_)

        # Split the data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

        # Build the model
        model = Sequential()
        model.add(Dense(10, input_shape=(num_features,), activation='relu', name='fc1'))
        model.add(Dense(10, activation='relu', name='fc2'))
        model.add(Dense(5, activation='softmax', name='output'))

        # Adam optimizer with learning rate of 0.001
        optimizer = Adam(lr=0.001)
        model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        print('Neural Network Model Summary: ')
        print(model.summary())

        # Train the model
        training_time_start = time.time()

        history = model.fit(X_train, y_train, verbose=2, batch_size=50, epochs=200)
        y_train_pred = model.predict_classes(X_train)
        classifir_accuracy, classifier_f1_score = classifer_metrics(np.argmax(y_train, axis=1), y_train_pred)
        training_accuracy_DNN[dataset].append(classifir_accuracy)
        training_f1_score_DNN[dataset].append(classifier_f1_score)
        # training_accuracy_DNN[dataset].append(history.history.get('accuracy')[-1]*100)

        training_time_end = time.time()
        training_time_DNN[dataset].append(training_time_end - training_time_start)

        # Test on unseen data
        testing_time_start = time.time()
        y_test_pred = model.predict_classes(X_test)
        classifier_accuracy, classifier_f1_score = classifer_metrics(np.argmax(y_test, axis=1), y_test_pred)
        testing_accuracy_DNN[dataset].append(classifier_accuracy)
        testing_f1_score_DNN[dataset].append(classifier_f1_score)
        # testing_accuracy_DNN[dataset].append(results[1]*100)
        testing_time_end = time.time()
        testing_time_DNN[dataset].append(testing_time_end - testing_time_start)


    print(f"\nAverage Training Time: --- {np.mean(training_time_DNN[dataset]):.4f} seconds ---")
    print(f"Average Training Accuracy: --- {np.mean(training_accuracy_DNN[dataset]):.2f}%")
    print(f"Average Training F1 Score: --- {np.mean(training_f1_score_DNN[dataset]):.2f}%")
    print(f"Average Testing Time: --- {np.mean(testing_time_DNN[dataset]):.4f} seconds ---")
    print(f"Average Testing Accuracy: --- {np.mean(testing_accuracy_DNN[dataset]):.2f}%")
    print(f"Average Testing F1 Score: --- {np.mean(testing_f1_score_DNN[dataset]):.2f}%\n")


for dataset in datasets:
    print("----------------------------------------------\n")
    print(f"------------- {dataset[:-4]} Summary -------------")
    for i in range(iterations):
        print(f"Training Time_{i+1}: --- {training_time_DNN[dataset][i]:.4f} seconds ---")
        print(f"Training Accuracy_{i+1}: --- {training_accuracy_DNN[dataset][i]:.2f}%")
        print(f"Training F1 Score_{i+1}: --- {training_f1_score_DNN[dataset][i]:.2f}%")
        print(f"Testing Time_{i+1}: --- {testing_time_DNN[dataset][i]:.4f} seconds ---")
        print(f"Testing Accuracy_{i+1}: --- {testing_accuracy_DNN[dataset][i]:.2f}%")
        print(f"Testing F1 Score_{i+1}: --- {testing_f1_score_DNN[dataset][i]:.2f}%")
        print("----------------------------------------------\n")

    print(f"\nAverage Training Time: --- {np.mean(training_time_DNN[dataset]):.4f} seconds ---")
    print(f"Average Training Accuracy: --- {np.mean(training_accuracy_DNN[dataset]):.2f}%")
    print(f"Average Training F1 Score: --- {np.mean(training_f1_score_DNN[dataset]):.2f}%")
    print(f"Average Testing Time: --- {np.mean(testing_time_DNN[dataset]):.4f} seconds ---")
    print(f"Average Testing Accuracy: --- {np.mean(testing_accuracy_DNN[dataset]):.2f}%")
    print(f"Average Testing F1 Score: --- {np.mean(testing_f1_score_DNN[dataset]):.2f}%\n")



print(f"------------------------- KNN -----------------------------")


#KNN
training_time_KNN = {k:[] for k in datasets}
training_accuracy_KNN = {k:[] for k in datasets}
training_f1_score_KNN = {k:[] for k in datasets}
testing_time_KNN = {k:[] for k in datasets}
testing_accuracy_KNN = {k:[] for k in datasets}
testing_f1_score_KNN = {k:[] for k in datasets}


for dataset in datasets:
    print(f"------------- {dataset[:-4]} Execution -------------")
    with open(f'./datasets/{dataset}','r') as csv_file:
        csv_reader = csv.reader(csv_file)
        data = list(csv_reader)

    data = np.array(data)
    data = data.astype(np.float)
        
    num_observations = data.shape[0]
    num_features = data.shape[1] - 1

    print(f"Dataset Size: {data.shape}\n")
    for i in range(iterations):
        # Shuffle the data
        shuffle_idx = np.random.permutation(num_observations)
        shuffled_data = data[shuffle_idx,:]

        X = shuffled_data[:,:-1]
        y = shuffled_data[:,-1]

        le = LabelEncoder()
        y = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

        # Instantiate learning model (k = 5)
        classifier = KNeighborsClassifier(n_neighbors=5)

        # Fitting the model
        training_time_start = time.time()

        classifier.fit(X_train, y_train)
        y_train_pred = classifier.predict(X_train)
        classifir_accuracy, classifier_f1_score = classifer_metrics(y_train, y_train_pred)
        training_accuracy_KNN[dataset].append(classifir_accuracy)
        training_f1_score_KNN[dataset].append(classifier_f1_score)

        training_time_end = time.time()
        training_time_KNN[dataset].append(training_time_end - training_time_start)

        # Predicting the Test set results
        testing_time_start = time.time()
        
        y_test_pred = classifier.predict(X_test)
        classifir_accuracy, classifier_f1_score = classifer_metrics(y_test, y_test_pred)
        testing_accuracy_KNN[dataset].append(classifir_accuracy)
        testing_f1_score_KNN[dataset].append(classifier_f1_score)
        
        testing_time_end = time.time()
        testing_time_KNN[dataset].append(testing_time_end - testing_time_start)

    print(f"\nAverage Training Time: --- {np.mean(training_time_KNN[dataset]):.4f} seconds ---")
    print(f"Average Training Accuracy: --- {np.mean(training_accuracy_KNN[dataset]):.2f}%")
    print(f"Average Training F1 Score: --- {np.mean(training_f1_score_KNN[dataset]):.2f}%")
    print(f"Average Testing Time: --- {np.mean(testing_time_KNN[dataset]):.4f} seconds ---")
    print(f"Average Testing Accuracy: --- {np.mean(testing_accuracy_KNN[dataset]):.2f}%")
    print(f"Average Testing F1 Score: --- {np.mean(testing_f1_score_KNN[dataset]):.2f}%\n")
