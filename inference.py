import os
from joblib import load
import numpy as np

# #############################################################################
# Load model
#print("Loading model from working directory")
clf = load('CustomerClassification.joblib') 
# #############################################################################
# Run inference
forecast = clf.forecast(steps=14)[0]
result = [] 
for i in forecast:
    print(np.exp(i))
    result.append(np.exp(i)*7)
print("Predict revenue...")
print(result)



y_pred=clf.predict(X_test)


X_NeedPredict=NeedPredict[['PRICE','FirstloginRange','Period','AveTrans','LAUNCH CAMPAIGN','MINOR EVENT SPONSORSHIP',
               'MAJOR ADVERTISING BUY', 'MINOR ADVERTISING BUY','MAJOR EVENT SPONSORSHIP']]

y_pred_problem2=clf.predict(X_NeedPredict)


NeedPredict['result']=y_pred_problem2

Result_NeedPredict=NeedPredict[['USERID','result']]
Result_NeedPredict.head()

Result= Result_NeedPredict.append(Result_DontPredict, ignore_index=True)
Result=Result.sort_values(by=['USERID'], ascending=True)


