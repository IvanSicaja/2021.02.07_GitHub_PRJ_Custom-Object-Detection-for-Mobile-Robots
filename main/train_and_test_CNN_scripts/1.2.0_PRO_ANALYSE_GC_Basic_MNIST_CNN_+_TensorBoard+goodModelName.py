# If we are working with 28x28 pixel image, final shape for sending in model need to be 1,28,28.
# 1,28,28 is different from 28,28,1, because in last case order means following height, width, number of channels.
# 1,28,28 is also different from 1,784 because in last case order means following height, width.

#  TODO: MAKE MODEL SIMPLE FOR RESEARCHING
#  TODO: SAVE TRAIN MODEL LEARNING CURVE IMAGE: ACCURACY, VALIDATION
#  TODO: DRAW BOUNDING BOX AROUND TESTING IMAGE
#  TODO: ADD DETAIL ANALYSE ACTIVATION OPTIONS
#  DONE: SAVE IMAGES IN EXCEL FILE WITH CORRESPONDING PIXELS VALUES
#  DONE: EXPLORE -> WHY 1,28,28 IS DIFFERENT FROM 1,784. WHY WE NEED TO SEND 1,28,28 TO THE MODEL, AND 28,28 ALSO INST POSSIBLE?

#----------> Import all needed modules <----------#
import os
import cv2
import datetime                  # Module used for print current date and time
import keyboard                  # Module used for waiting for defined key pressing
import numpy as np
import pandas as pd              # Module used for save image shape with every pixel value in .CSV fromat
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def printImagesShapes (img,imgName):

    print(imgName+ ' shape is: ', img.shape)


#----------> Introduction console text <----------#
print("-------------------------------")
print("Welcome to Ultimate CNN script!")                                        # Print welcome text in the console
print("-------------------------------\n")
print("FOR TRAN MODEL PRESS 'Left Key' <=>  FOR TEST MODEL PRESS 'Right Key'")  # Decide if to load an existing model or to train a new one
while True:                                                                     # Make while loop for make train or text decision
    if keyboard.is_pressed('left'):                                             # If key 'left' is pressed
        print("You pressed 'Left key'!", end='')
        print("TRAIN BEGINS.")
        train_new_model=True
        break                                                                   # Break the loop

    if keyboard.is_pressed('right'):                                            # If key 'right' is pressed
        print("You pressed 'Right key'!", end='')
        print("TEST BEGINS.")
        train_new_model=False
        break                                                                   # Break the loop

############################################### TRAINING MODEL - CODE BELLOW ###############################################
if train_new_model:
    # ----------> Get current time <----------#
    currentTimeAndDate = datetime.datetime.now()                                # Get current date and time
    # print("Current date and time is: ")
    # print(currentTimeAndDate.strftime("%y-%m-%d_%H:%M:%S"))
    currentTimeAndDate=currentTimeAndDate.strftime("%y-%m-%d__%Hh-%Mm-%Ss")     # Get current date and time in order to make unique saved names

    # ----------> Set up Tensorboard i.e. create tb callback<----------#
    # NOTE: Needed to add "tensorboardCallback" variable bellow in model.fit(callbacks=[tensorboardCallback])
    # NOTE: If you want to see TensorBoard visualisation you need to go to the parent directory
    # NOTE: of the log_dir with command prompt and the run the command -> "tensorboard --logdir 'here enter name of your log_dir folder'"
    tensorboardCallback=tf.keras.callbacks.TensorBoard(log_dir='../4.0_developmental_phase_CNN_models/2.0_MNIST/MNIST__'+str(currentTimeAndDate)+'__', histogram_freq=1)

    # ----------> Loading the 2.0_MNIST data set with samples and splitting it <----------#
    mnist = tf.keras.datasets.mnist
    # NOTE: X data row is single row of all pixels of the image, Y data is labeled value
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    printImagesShapes(X_train,'X_train')
    printImagesShapes(X_test,'X_test')
    printImagesShapes(Y_train,'Y_train')
    printImagesShapes(Y_test,'Y_test')

    # ----------> Visualize number of instances in every class from dataset <----------#
    # NOTE: Class number of letter A= class 10,B=class 11... Classes 0-9 are numbers 0-9
    plt.figure(figsize=(15, 7))                                                    # Define graph size
    sns.set_theme(style="darkgrid")                                                # Define graph style
    # NOTE: sns.countplot(Y_train, color='skyblue' ) -> color= matplotlib names
    g = sns.countplot(Y_train, color='cornflowerblue')                             # Plot number of instances per class from Y_train
    plt.title("Visualized number of instances in every class from dataset")
    plt.xlabel('Class name')
    plt.ylabel('Number of instances per class')
    plt.show()  # Show defined graph
    # NOTE: We don't use following function, but it is amazing function for data analyse
    #Y_train.value_counts()  # Function which counts how many times tha same data is appearing in the defined dataset

    # ----------> See and save values of every pixel for specified dataset <----------#
    # NOTE: 1 row = 1 image, we want to read only one row from our dataset shape (60000, 28, 28)
    # print("Selected image pixel values are: ",X_train[1,:,:])                   # If you want print pixel values of a image from train database uncomment this line

    # ----------> Save pixel values in .CSV file because of intuitive visualization <----------#
    pd.DataFrame(X_train[1,:,:]).to_csv("../../developmental_phase_images/1.3_images_converted_in_.CSV/1.0_Visualised_one_dataset_image/visualized_unknown_dataset_image.csv")  # Save image in .CSV file

    # ----------> Normalizing the data pixel vales <----------#
    # NOTE: axis=1 do operations on rows, axis=0 do operations on columns
    # NOTE: axis=1 is default for this model (tf keras ecosystem), but it can be different values for others models -> e.g. axis=-1 is default value for the RNN model
    X_train = tf.keras.utils.normalize(X_train, axis=1)
    X_test = tf.keras.utils.normalize(X_test, axis=1)

    # ----------> Create a neural network model <----------#
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())                                  # Add one flattened input layer for the pixels
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))    # Add dense hidden layers
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))    # Add dense hidden layers
    model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))  # Add one dense output layer for the 10 digits

    # ----------> Model learning curve definition and optimization <----------#
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # ----------> Model training parameters <----------#
    epochs=1                                                              # Number of training epochs
    validation_split=0.1                                                  # E.g. validation_split=0.1, means use 90% train model images for train and last 10% for validation
    batch_size=32                                                         # Number of images which we send for training at the same time

    # ----------> Train model with defined parameters <----------#
    # NOTE: history isn't necessary, but it can be used for access to model metrics like accuracy, val_accuracy i.e.
    history=model.fit(X_train, Y_train, validation_split=validation_split, epochs=epochs, batch_size=batch_size, callbacks=[tensorboardCallback])

    # ----------> Get and print model validation and training accuracy <----------#
    val_loss, val_acc = model.evaluate(X_test, Y_test)
    #print(val_loss)                                                     # Not needed, used "history" object for printing key model metrics
    #(val_acc)                                                           # Not needed, used "history" object for printing key model metrics
    # print('History.history: ',history.history)                         # Print key metrics of model
    training_accuracy=history.history['accuracy'][-1]                    # Store training accuracy in a variable
    validation_accuracy=history.history['val_accuracy'][-1]              # Store validation accuracy in a variable
    print("Model training accuracy:   ",training_accuracy)               # Print training accuracy of model
    print("Model validation accuracy: ", validation_accuracy)            # Print validation accuracy of model

    # ----------> Plotting models learning curves: train-accuracy and validation-accuracy <----------#
    # NOTE: This isn't used because we use TensorBoard for visualisation
    # NOTE: At least 2 epochs need to be done in model training if you want print it in this way
    # plt.plot(history.history['accuracy'])                              # Add train-accuracy metric on the graph
    # plt.plot(history.history['val_accuracy'])                          # Add validation-accuracy metric on the graph
    # plt.title('Model accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['Train acc', 'Validation acc'], loc='upper left')
    # plt.show()

    # ----------> Plot the confusion matrix  <----------#
    # NOTE: Confusion matrix makes sense only for validation data
    # NOTE: For confusion matrix creation we need two 1 dimensional arrays which only contains a index of predicted class
    # NOTE: One array for true test-validation classes, another array for predicted classes from test-validation samples
    print('X_test shape is:',X_test.shape)
    print('Y_test shape is:',Y_test.shape)


    Y_pred = model.predict(X_test)                                              # Predict the values from the validation dataset
    print('Y_pred shape is:', Y_pred.shape)
    Y_pred_classes = np.argmax(Y_pred,axis=-1)                                  # Convert predictions classes to one hot vectors
    print('Y_pred_classes:', Y_pred_classes.shape)
    Y_true = Y_test                                                             # Convert validation observations to one hot vectors if it is needed
    print('Y_true shape is:', Y_true.shape)

    # NOTE: Print some values form Y_true and Y_predicted_classes because it is
    # NOTE: final to see do we have values in the class range e.g. for MNIST values need to be in range 0-9
    print('Y_pred_classes values is:', Y_pred_classes )                         # Print values of array
    print('Y_true values is:', Y_true )                                         # Print values of array
    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes,normalize='true')   # Compute the confusion matrix, NOTE: Normalize parameter normalizes confusion matrix -> values in percentage

    # Plot the confusion matrix
    # NOTE: sns.heatmap is a graph which changes color saturation according to number value
    f, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(confusion_mtx, annot=True, linewidths=0.01, cmap="Blues", linecolor="gray", fmt='.3f', ax=ax, square=True ) # NOTE: "fmt" parameter defines numbers of decimal points for CM percentage
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

    # ----------> Save model with robust name  <----------#
    # Saving the model
    model.save('../4.0_developmental_phase_CNN_models/2.0_MNIST/MNIST__Val-acc='+str(round(validation_accuracy,4))+'__Train-acc='+str(round(training_accuracy,4))+'__'+str(currentTimeAndDate)+'__Epochs='+str(epochs)+'__Batch-size='+str(round(batch_size))+'__'+'.h5')

    # ----------> Get trained model summary <----------#
    model.summary()

############################################### TEST TRAINED MODEL - CODE BELLOW ###############################################
else:

    # ----------> Define test images directory and wanted trained model <----------#
    custom_images_for_test_dir = '../../developmental_phase_images/1.2_fully_prepared_28x28_images_for_test_CNN_model'                                  # Directory of images for test our model
    model = tf.keras.models.load_model('../../4.0_Good_saved_CNN_models/1.0_MNIST/MNIST__Val-acc=0.9752__Train-acc=0.9773__22-08-27__11h-05m-51s__Epochs=3__Batch-size=32__.h5')  # Load the model

    # ----------> Load custom images and predict them <----------#
    while True:
        # NOTE: If you want to test model without all preprocessing just send image from tets data :)
        img_raw = cv2.imread(custom_images_for_test_dir+'/digit1.png')[:,:,0]                                               # Load raw image as grayscale -> Shape (28,28,1) -> (height, width, chanel number)
        print("img_raw shape: ", img_raw.shape)                                                                            # Print image shape
        #print(img_raw)                                                                                                    # Shows every image pixel value
        #height, width=img_raw.shape                                                                                       # Possible to read separately height, width, channels = img.shape, and then print it

        # ----------> Reshape loaded test image <----------#
        img_raw_reshaped = np.reshape(img_raw, (img_raw.shape[0], img_raw.shape[1]))                                       # Reshaping image, order: image array, img height, img width
        print("img_raw_reshaped shape: ", img_raw_reshaped.shape)                                                          # Print image shape
        pd.DataFrame(img_raw_reshaped).to_csv("../../developmental_phase_images/1.3_images_converted_in_.CSV/img_raw_reshaped_converted_in_28x28_csv.csv")  # Save image in .CSV file

        # ----------> Invert colours of reshaped test image <----------#
        img_final = np.invert(np.array([img_raw]))                                                                         # Inverting image pixels value e.g: black color to white or reverse
        print('img_final', img_final.shape)                                                                                # Print image shape
        #print('img_final',img_final)                                                                                      # Shows every image pixel value
        #height, width = img_final.shape                                                                                   # Possible to read separately height, width, channels = img.shape, and then print it
        # plt.imshow(img_final[0])                                                                                         # Define image, and displaying way  OLD ORDER-> plt.imshow(img_final[0], cmap=plt.cm.binary)
        # plt.title("Final image, with batchsize")                                                                         # Write image title
        # plt.show()                                                                                                       # Display image on the screen

        # NOTE: IMAGE HAS DIFFERENT SHAPE AFTER np.invert() function : (28,28) -> (1,28,28). (1,28,28) is sent to CNN input, because that is wanted input.
        # NOTE: (1,28,28) -> (batch-size, height, width)
        # NOTE: (28,28,1) -> (height, width, chanel number) -> chanel number can be: R,G,B,opacity...

        # ----------> Add or remove dimensions of inverted colour image if it is needed <----------#
        img_final_reshaped = np.reshape(img_final, (img_final.shape[1],img_final.shape[2]))                                # Reshaping image, order: image array, img height, img width
        print("img_final_reshaped shape: ", img_final_reshaped.shape)                                                      # Print image shape
        print("img_final_reshaped shape values: ",img_final_reshaped)                                                      # Shows every image pixel value

        # ----------> Save pixel values in .CSV file because intuitive visualization <----------#
        pd.DataFrame(img_final_reshaped).to_csv( "../../developmental_phase_images/1.3_images_converted_in_.CSV/img_final_reshaped_converted_in_28x28_csv.csv")  # Save image in .CSV file

        # ----------> Predict value on the image and print it <----------#
        prediction = model.predict(img_final)                                                                              # Sending image to trained model
        print("The number is probably a {}".format(np.argmax(prediction)))                                                 # Printing prediction value and spent time

        # ----------> For the program exit press key <----------#
        if keyboard.is_pressed('esc'):                                                                                     # Key press event  NOTE -> Just change parameter for another key event keyboard.is_pressed('0')
            print("CLOSING PROGRAM...", end='')
            break                                                                                                          # Break while True loop

print("THE END :)")


