# If we are working with 28x28 pixel image, final shape for sending in model need to be 1,28,28.
# 1,28,28 is different from 28,28,1, because in last case order means following height, width, number of channels.
# 1,28,28 is also different from 1,784 because in last case order means following height, width.

#  TODO: MAKE MODEL SIMPLE FOR RESEARCHING
#  TODO: SAVE TRAIN MODEL LEARNING CURVE IMAGE: ACCURACY, VALIDATION
#  TODO: DRAW BOUNDING BOX AROUND TESTING IMAGE
#  TODO: ADD DETAIL ANALYSE ACTIVATION OPTIONS


#  DONE: SAVE IMAGES IN EXCEL FILE WITH CORRESPONDING PIXELS VALUES
#  DONE: EXPLORE -> WHY 1,28,28 IS DIFFERENT FROM 1,784. WHY WE NEED TO SEND 1,28,28 TO THE MODEL, AND 28,28 ALSO INST POSSIBLE?

import os
import cv2
import time                      # Module used for waiting for making time delays
import keyboard                  # Module used for waiting for defined key pressing
import numpy as np
import pandas as pd              # Module used for save image shape with every pixel value in .CSV fromat
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn






# Directory paths
custom_images_for_test_dir= '../1.2_fully_prepared_28x28_images_for_test_CNN_model/'


print("-------------------------------")
print("Welcome to Ultimate CNN script!")
print("-------------------------------\n")


# Decide if to load an existing model or to train a new one

print("FOR TRAN MODEL PRESS 'ENTER' <=>  FOR TEST MODEL PRESS 'CTRL'")
while True:  # making a loop

    if keyboard.is_pressed('enter'):  # if key 'enter' is pressed
        print("You pressed 'Enter key'!", end='')
        print("TRAIN BEGINS.")
        train_new_model=True
        break  # finishing the loop

    if keyboard.is_pressed('ctrl'):  # if key 'enter' is pressed
        print("You pressed 'Ctrl key'!", end='')
        print("TEST BEGINS.")
        train_new_model=False
        break  # finishing the loop


############################################### TRAINING MODEL - CODE BELLOW ###############################################


if train_new_model:
    # Loading the 2.0_MNIST data set with samples and splitting it
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalizing the data (making length = 1)
    X_train = tf.keras.utils.normalize(X_train, axis=1)
    X_test = tf.keras.utils.normalize(X_test, axis=1)

    # Create a neural network model
    # Add one flattened input layer for the pixels
    # Add two dense hidden layers
    # Add one dense output layer for the 10 digits
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

    # Compiling and optimizing model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Training parameters
    epochs=2 # Number of training epochs

    # Training the model
    history=model.fit(X_train, y_train, validation_split=0.1, epochs=epochs)

    # Evaluating the model
    val_loss, val_acc = model.evaluate(X_test, y_test)
    #print(val_loss) # Not needed, used "history" object for printing key model metrics
    #(val_acc)       # Not needed, used "history" object for printing key model metrics


    # print('History.history: ',history.history)                # Print key metrics of model
    training_accuracy=history.history['accuracy'][-1]           # Store training accuracy in a variable
    validation_accuracy=history.history['val_accuracy'][-1]     # Store validation accuracy in a variable
    print("Model training accuracy:   ",training_accuracy)      # Print training accuracy of model
    print("Model validation accuracy: ", validation_accuracy)   # Print validation accuracy of model

    # Plotting models learning curves: train-accuracy and validation-accuracy
    # --> 1 epochs need to be done in model training if you want print it in this way
    plt.plot(history.history['accuracy'])        # Add train-accuracy metric on the graph
    plt.plot(history.history['val_accuracy'])    # Add validation-accuracy metric on the graph
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train acc', 'Validation acc'], loc='upper left')
    plt.show()

    # Saving the model
    model.save('../4.0_developmental_phase_CNN_models/handwritten_digits__TRAINING-ACCURACY='+str(round(training_accuracy,4))+'__VALIDATION-ACCURACY='+str(round(validation_accuracy,4))+'__EPOCHS='+str(epochs)+'_'+'.h5')
    model.summary()


############################################### TEST TRAINED MODEL - CODE BELLOW ###############################################

else:
    # Load the model
    model = tf.keras.models.load_model('../4.0_developmental_phase_CNN_models/handwritten_digits__TRAINING-ACCURACY=0.9651__VALIDATION-ACCURACY=0.9663__EPOCHS=2_.h5')

    # Load custom images and predict them
    while True:

        img_raw = cv2.imread(custom_images_for_test_dir+'digit1.png')[:,:,0]   # Load raw image as grayscale -> Shape (28,28,1) -> (height, width, chanel number)
        print("img_raw shape: ", img_raw.shape)                                # Print image shape
        #print(img_raw)                                                        # Shows every image pixel value
        #height, width=img_raw.shape                                           # Possible to read separately height, width, channels = img.shape, and then print it

        img_raw_reshaped = np.reshape(img_raw, (img_raw.shape[0], img_raw.shape[1]))                                       # Reshaping image, order: image array, img height, img width
        print("img_raw_reshaped shape: ", img_raw_reshaped.shape)                                                          # Print image shape
        pd.DataFrame(img_raw_reshaped).to_csv(
            "../../developmental_phase_images/1.3_images_converted_in_.CSV/img_raw_reshaped_converted_in_28x28_csv.csv")  # Save image in .CSV file

        img_final = np.invert(np.array([img_raw]))                # Inverting image pixels value e.g: black color to white or reverse
        print('img_final', img_final.shape)                       # Print image shape
        print('img_final',img_final)                              # Shows every image pixel value
        #height, width = img_final.shape                          # Possible to read separately height, width, channels = img.shape, and then print it
        # plt.imshow(img_final[0])                                  # Define image, and displaying way  OLD ORDER-> plt.imshow(img_final[0], cmap=plt.cm.binary)
        # plt.title("Final image, with batchsize")                  # Write image title
        # plt.show()                                                # Display image on the screen

        ##############
        # IMAGE HAS DIFFERENT SHAPE AFTER np.invert() function : (28,28) -> (1,28,28). (1,28,28) sent to NN input, because that is wanted input.
        # (1,28,28) -> (batch-size, height, width)
        # (28,28,1) -> (height, width, chanel number) -> chanel number can be: R,G,B,opacity...
        ##############


        img_final_reshaped = np.reshape(img_final, (img_final.shape[1],img_final.shape[2]))                                    # Reshaping image, order: image array, img height, img width
        print("img_final_reshaped shape: ", img_final_reshaped.shape)                                                          # Print image shape
        print("img_final_reshaped shape: ",img_final_reshaped)                                                                 # Shows every image pixel value
        pd.DataFrame(img_final_reshaped).to_csv(
            "../../developmental_phase_images/1.3_images_converted_in_.CSV/img_final_reshaped_converted_in_28x28_csv.csv")  # Save image in .CSV file


        prediction = model.predict(img_final)                               # Sending image to trained model
        print("The number is probably a {}".format(np.argmax(prediction)))  # Printing prediction value and spent time

        if keyboard.is_pressed('ctrl'):           # Key press event  NOTE -> Just change parameter for another key event keyboard.is_pressed('0')
            print("CLOSING PROGRAM...", end='')
            break                                 # Break while True loop

print("THE END :)")


