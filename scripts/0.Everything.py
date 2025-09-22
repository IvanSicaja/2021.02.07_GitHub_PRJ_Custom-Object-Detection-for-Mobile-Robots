import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# ----------> Import out dataset <----------#
train = pd.read_csv("../../3.0_Datasets_for_CNN_training/1.0_Simple_datasets/1.0_Labeled_20_classes_for_seaborn_graph_practice.csv")

Y_train = train["label"]                                    # Store entire "label" column defined in the .CSV file in Y_train variable
X_train = train.drop(labels=["label"], axis=1)
# ----------> Plot some samples <----------#
img = X_train.iloc[0].to_numpy()                            # .iloc[0] pandas function which can access to any cell, row or column from excel -> EXPLANATION: iloc[0] return wanted row, iloc[0,1] = return wanted cell, iloc[:,3] return wanted column
print("#############################################################################")
print("BEFORE RESHAPE SHAPES")
print("Train img shape is :", img.shape)
print("#############################################################################")
img = img.reshape((28, 28))
plt.imshow(img,cmap='gray')
print(plt.title(train.iloc[0,0]))  # Get name of the column, in our case name of the column is "label"
plt.axis("off")
plt.show()