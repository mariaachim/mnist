import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.datasets import mnist
from sklearn.metrics import confusion_matrix
import seaborn as sns
import random

class NeuralNetwork:
    def __init__(self, num_classes):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        self.num_classes = num_classes
        np.random.seed(0)
        print(self.x_train.shape, self.y_train.shape)
        print(self.x_test.shape, self.y_test.shape)
    def prepareTrainingData(self):
        self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes)
        self.y_test = keras.utils.to_categorical(self.y_test, self.num_classes)

        # normalise data
        self.x_train = self.x_train / 255.0
        self.x_test = self.x_test / 255.0

        # reshape data
        self.x_train = self.x_train.reshape(self.x_train.shape[0], -1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], -1)
        print("training data prepared")
    def create(self):
        self.model = Sequential()
        self.model.add(Dense(units=128, input_shape=(784,), activation='relu'))
        self.model.add(Dense(units=128, activation='relu'))
        self.model.add(Dropout(0.25)) # 25% of neurons are deactivated to prevent overfitting from network
        self.model.add(Dense(units=10, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # categorical crossentropy is used when dealing with multiple classes

        self.model.summary()
        print("model created")
    def train(self):
        batch_size = 512
        epochs = 10
        self.model.fit(x = self.x_train, y = self.y_train, batch_size = 512, epochs = 10)
        test_loss, test_acc = self.model.evaluate(self.x_test, self.y_test)
        print("Test Loss: {}, Test Accuracy: {}".format(test_loss, test_acc))
    def predictImages(self):
        f, ax = plt.subplots(1, self.num_classes, figsize=(20, 20))
        for i in range(self.num_classes):
            sample = self.x_train[self.y_train == i][0]
            ax[i].imshow(sample, cmap = 'gray')
            ax[i].set_title("Label: {}".format(i), fontsize=16)
    def evaluate(self):
        test_loss, test_acc = self.model.evaluate(self.x_test, self.y_test)
        print("Test Loss: {}, Test Accuracy: {}".format(test_loss, test_acc))

        self.y_pred = self.model.predict(self.x_test)
        y_pred_classes = np.argmax(self.y_pred, axis=1)
        print(self.y_pred)
        print(y_pred_classes)
    def predictSingle(self):
        self.y_pred = self.model.predict(self.x_test)
        self.y_pred_classes = np.argmax(self.y_pred, axis=1)
        random_idx = random.randint(0, len(self.x_test) - 1)
        x_sample = self.x_test[random_idx]
        print(self.y_test)
        self.y_true = np.argmax(self.y_test, axis=1)   #### error here
        print(self.y_true)     
        y_sample_true = self.y_true[random_idx]
        y_sample_pred_class = self.y_pred_classes[random_idx]

        fig, ax = plt.subplots(figsize=(20, 20))
        ax.set_title("Predicted: {}, Actual: {}".format(y_sample_pred_class, y_sample_true), fontsize=16)
        ax.imshow(x_sample.reshape(28, 28), cmap='gray')

        self.model.summary()

    def confusionMatrix(self):
        confusion_mtx = confusion_matrix(self.y_true, self.y_pred_classes)

        fig, ax = plt.subplots(figsize = (15, 10))
        ax = sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap="Greens")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix")

    def showErrors(self):
        # investigating errors
        errors = (self.y_pred_classes - self.y_true != 0) # predicted classes are not the same as actual classes
        y_pred_classes_errors = self.y_pred_classes[errors]
        y_pred_errors = self.y_pred[errors]
        y_true_errors = self.y_true[errors]
        x_test_errors = self.x_test[errors]

        y_pred_errors_probability = np.max(y_pred_errors, axis=1)
        true_probability_errors = np.diagonal(np.take(y_pred_errors, y_true_errors, axis=1))
        diff_errors_pred_true = y_pred_errors_probability - true_probability_errors

        # get list of indices of sorted differences
        sorted_idx_diff_errors = np.argsort(diff_errors_pred_true)
        top_idx_diff_errors = sorted_idx_diff_errors[-5:] # last 5

        # show top errors
        num = len(top_idx_diff_errors)
        f, ax = plt.subplots(1, num, figsize=(30,30))

        for i in range(0, num):
            idx = top_idx_diff_errors[i]
            sample = x_test_errors[idx].reshape(28,28)
            y_t = y_true_errors[idx]
            y_p = y_pred_classes_errors[idx]
            ax[i].imshow(sample, cmap='gray')
            ax[i].set_title("Predicted Label: {}\nTrue Label: {}".format(y_p, y_t), fontsize=12)


nn = NeuralNetwork(10)
nn.create()
nn.predictImages()
nn.prepareTrainingData()
nn.train()
nn.evaluate()
nn.predictSingle()
nn.confusionMatrix()
nn.showErrors()

plt.show()