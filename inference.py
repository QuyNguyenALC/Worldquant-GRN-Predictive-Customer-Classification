import os
from joblib import load
import numpy as np
import pandas as pd

# #############################################################################
# Load model
#print("Loading model from working directory")
clf = load('CustomerClassification.joblib') 
# #############################################################################
# Run inference	
X_NeedPredict=pd.read_csv("data/X_NeedPredict.csv")

y_pred_problem2=clf.predict(X_NeedPredict)
print("The predict is: ")
print(y_pred_problem2)
