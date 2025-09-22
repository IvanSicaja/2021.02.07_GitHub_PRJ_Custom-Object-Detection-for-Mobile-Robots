# import os                         # Can be useful for preforming command prompt orders, change directories, list all file in directory
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# filter warnings
warnings.filterwarnings('ignore')

# ----------> Import out dataset <----------#
train = pd.read_csv("../../3.0_Datasets_for_CNN_training/1.0_Labeled_PRO__My_Data_A-Z_and_0-9__HandwrittenData_432000+_.csv")

# ----------> Print dataset shape <----------#
print("#############################################################################")
print("Initial¸train shape is :", train.shape)
print("#############################################################################")
train.head()

# ----------> Import dataset for testing our trained model <----------#
# NOTE: This dataset isn't used for any kind of training model
# NOTE: It is easier to use for the test finished model then take, preprocess, reshape custom images
test = pd.read_csv("../../3.0_Datasets_for_CNN_training/2.0_Unlabeled_PRO__A_Z_HandwrittenData_370000+_.csv")
print("#############################################################################")
print("Initial¸test shape is :", test.shape)
print("#############################################################################")
test.head()

# ----------> Separate images from images labeled values <----------#
# NOTE: axis=0 <-> axis='index'<-> -y axis  and means do operations along the -y axis -> get wanted row
# NOTE: axis=1 <-> axis='column'<-> x axis  and means do operations along the x axis -> get wanted column
Y_train = train["label"]                                    # Store entire "label" column defined in the .CSV file in Y_train variable
X_train = train.drop(labels=["label"], axis=1)              # Drop 'label' column from dataset .CSV file -> only images left in X_train
print("#############################################################################")
print("LABELS AND IMAGES IS SEPARATED - SHAPES")
print("Train label shape is :", Y_train.shape)
print("Train image shape is :", X_train.shape)
print("#############################################################################")

# ----------> Visualize number of instances in every class from dataset <----------#
# NOTE: Class number of letter A=10,B=11... Classes 0-9 are numbers 0-9
plt.figure(figsize=(15,7))                                  # Define graph size
sns.set_theme(style="darkgrid")                             # Define graph style
g = sns.countplot(Y_train, color='cornflowerblue')          # Plot number of instances per class from Y_train
plt.title("Visualized number of instances in every class from dataset")
plt.xlabel('Class name')
plt.ylabel('Number of instances per class')
plt.show()                                                  # Show defined graph
# NOTE: We don't use following function, but it is amazing function for data analyse
Y_train.value_counts()                                      # Function which counts how many times tha same data is appearing in the defined dataset

# ----------> Plot some samples <----------#
img = X_train.iloc[0].to_numpy()                            # .iloc[0] pandas function which can access to any cell, row or column from excel -> EXPLANATION: iloc[0] return wanted row, iloc[0,1] = return wanted cell, iloc[:,3] return wanted column
print("#############################################################################")
print("BEFORE RESHAPE SHAPES")
print("Train img shape is :", img.shape)
print("#############################################################################")
img = img.reshape((28, 28))
# plt.imshow(img,cmap='gray')
# plt.title(train.iloc[0,0])  # Get value from the selected column, it can be every data type: string, int...
# plt.axis("off")
# plt.show()
print("#############################################################################")
print("PREVIEWED SHAPES")
print("Train img shape is :", img.shape)
print("#############################################################################")

# ----------> Plot some samples <----------#
img = X_train.iloc[3].to_numpy() # Get third row from X_train
img = img.reshape((28, 28))
# plt.imshow(img,cmap='gray')
# plt.title(train.iloc[3,0])
# plt.axis("off")
# plt.show()

# ----------> Normalize the data <----------#
X_train = X_train / 255.0
test = test / 255.0
print("#############################################################################")
print("AFTER NORMALISE SHAPES")
print("x_train shape: ", X_train.shape)
print("test shape: ", test.shape)
print("#############################################################################")

# ----------> Reshape the data <----------#
# NOTE "-1" in reshape(-1, 28, 28, 1) means "Give me the whenever the shape is, and I will from that shape create shape (28, 28, 1)"
X_train = X_train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)
print("#############################################################################")
print("AFTER RESHAPEING BY (-1,28,28,1)")
print("x_train shape: ", X_train.shape)
print("test shape: ", test.shape)
print("#############################################################################")

# ----------> Convert numpy array or vector in binary categorical form <----------#
# EXAMPLE: class_vector =[2, 5, 6, 1, 4, 2, 3, 2]->output_matrix = to_categorical(class_vector, num_classes = 7, dtype ="int32")
# Output -> [[0 0 1 0 0 0 0]
#           [0 0 0 0 0 1 0]
#           [0 0 0 0 0 0 1]
#           [0 1 0 0 0 0 0]
#           [0 0 0 0 1 0 0]
#           [0 0 1 0 0 0 0]
#           [0 0 0 1 0 0 0]
#           [0 0 1 0 0 0 0]]
from keras.utils.np_utils import to_categorical
Y_train = to_categorical(Y_train, num_classes=36)

# ----------> Split the input dataset for the train and the validation set, needed for training -> "fit" method <----------#
from sklearn.model_selection import train_test_split
# NOTE: X_train = images for training model
# NOTE: Y_train = classes of X_train images for training model
# NOTE: X_val = images for validating model
# NOTE: Y_val = classes of X_val images for validating model
# NOTE: test_size=0.3 divides your dataset 30% for validation, 70% for training
# NOTE: We don't want to take first 70% of dataset for training because
# NOTE: because maybe in the last 30% od data set we have another classes which don't exist in first 70% of dataset
# NOTE: Our dataset will be every time divided with random samples if we don't specify the random state value
# NOTE: Unspecified value of random_state parameter can result with different model accuracy for exactly the same model
# NOTE: If we define random_state=1 every time our test and validation data will contain the same examples
# NOTE: for each self, until we don't change random_state parameter to another int value
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.3, random_state=2)

# NOTE: Now X_train, X_val, Y_train, Y_val now have different size values
print("#############################################################################")
print("AFTER SPLITTING FOR VALIDATION")
print("x_train shape", X_train.shape)
print("x_test shape", X_val.shape)
print("y_train shape", Y_train.shape)
print("y_test shape", Y_val.shape)
print("#############################################################################")

# ----------> Plot some samples <----------#
# plt.imshow(X_train[2][:,:,0],cmap='gray')
# plt.show()

from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# ----------> Create Convolutional Neural Network, CNN <----------#
model = Sequential()
#
model.add(Conv2D(filters=8, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
#
model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))
# fully connected
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(36, activation="softmax"))

# Define the optimizer
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

# Compile the model
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

##################################################################################################################################
epochs = 60  # for better result increase the epochs
batch_size = 512
##################################################################################################################################

# ----------> Data augmentation <----------#
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # dimesion reduction
    rotation_range=False,  # randomly rotate images in the range 5 degrees
    zoom_range=1,  # Randomly zoom image 5%
    width_shift_range=1,  # randomly shift images horizontally 5%
    height_shift_range=1,  # randomly shift images vertically 5%
    horizontal_flip=False,  # randomly flip images
    vertical_flip=True)  # randomly flip images

datagen.fit(X_train)

# Fit the model
history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                              epochs=epochs, validation_data=(X_val, Y_val),
                              steps_per_epoch=X_train.shape[0] // batch_size)

model.summary()

# Plot the loss and accuracy curves for training and validation
plt.plot(history.history['val_loss'], color='b', label="validation loss")
plt.title("Test Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# confusion matrix
import seaborn as sns

# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors
Y_pred_classes = np.argmax(Y_pred, axis=1)
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val, axis=1)
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

# plot the confusion matrix
f, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01, cmap="Greens", linecolor="gray", fmt='.1f', ax=ax)
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.title("Confusion Matrix")
# plt.show()


# Saving model
model.save(
    "../../models/trained/developmental/Model_Batchsize=" + str(batch_size) + "__Epoch=" + str(epochs) + "__Accuracy=.h5")
print("Model saved.")

############################################### TEST TRAINED MODEL - CODE BELLOW ###############################################

# ----------> Test trained model with a row from dataset imported for testing  <----------#
image_index = 77777                                                         # Wanted row from dataset

# ----------> Display wanted row i.e. image for test  <----------#
print("################################################################")
print("Test image shape is :", test[image_index][:, :, 0].shape)
print("################################################################")
print("Plotting image for prediction")
plt.imshow(test[image_index][:, :, 0], cmap='gray')
plt.show()
print("Image for predivtion plotted")
print("################################################################")

# ----------> Predict value from wanted row i.e. image for test  <----------#
pred = model.predict(test[image_index].reshape(1, 28, 28, 1))
print("Predicted value is : ")
print(pred.argmax())
# plt.show()

# print("                                                                ")
print("################################################################")

print("End of script ;)")
