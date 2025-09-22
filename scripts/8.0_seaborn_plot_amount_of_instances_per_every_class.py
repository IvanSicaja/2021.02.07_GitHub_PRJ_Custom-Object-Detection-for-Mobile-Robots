# import os                         # Can be useful for preforming command prompt orders, change directories, list all file in directory
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# filter warnings
warnings.filterwarnings('ignore')

# ----------> Import out dataset <----------#
train = pd.read_csv("../../3.0_Datasets_for_CNN_training/1.0_Simple_datasets/1.0_Labeled_20_classes_for_seaborn_graph_practice.csv")

# ----------> Print dataset shape <----------#
print("#############################################################################")
print("Initial¸train shape is :", train.shape)
print("#############################################################################")
train.head()

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
plt.figure(figsize=(15,7))
sns.set_theme(style="darkgrid")
print('Y_train shape:\n',Y_train)
g = sns.countplot(Y_train)
plt.title("Visualized number of instances in every class from dataset")
plt.xlabel('Class name')
plt.ylabel('Number of instances per class')
plt.show()
Y_train.value_counts()