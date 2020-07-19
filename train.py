print("Problem 2: Customer classification")
print("start...")


import matplotlib.pylab as plt
# %matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
import requests, pandas as pd, numpy as np
from pandas import DataFrame
from io import StringIO
import time, json
from datetime import date, datetime
import datetime
import statsmodels
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from joblib import dump, load


print("Import data")
activity= pd.read_csv('data/activity.csv')
marketing=pd.read_csv('data/marketing.csv')
pricing=pd.read_csv('data/pricing.csv')
transaction=pd.read_csv('data/transaction.csv')


print("Feature engineering")
activity.DATE = pd.to_datetime(activity.DATE, format="%Y%m%d")
marketing.DATE = pd.to_datetime(marketing.DATE, format="%Y%m%d")
pricing.DATE = pd.to_datetime(pricing.DATE, format="%Y%m%d")
transaction.DATE = pd.to_datetime(transaction.DATE, format="%Y%m%d")


Loggedin=activity.loc[activity['ACTIVITY'] == 'logged in']
CreateAcc=activity.loc[activity['ACTIVITY'] == 'created account']


#FirstLoggin=Loggedin[Loggedin['DATE'].isin(Loggedin.groupby('USERID').min()['DATE'].values)]
FirstLoggin=Loggedin.loc[Loggedin.groupby('USERID')['DATE'].idxmin()]


#We need to create data with label, Because in this problem. We will need to classification of some users, which are
#create and loggedin the first time in month X and classification base on their transction in next 3 months (X+1, X+2,X+3)
FirstLogginBefore20160403=FirstLoggin.loc[FirstLoggin.DATE<'2016-07-02']


#we will find for each user, how long from create account to first loggin.
AccountInfo=pd.merge(CreateAcc,FirstLoggin, on=['USERID'], how='left')
AccountInfo.fillna(0)
AccountInfo['FirstloginRange']=(AccountInfo['DATE_y']-AccountInfo['DATE_x']).astype('timedelta64[D]')+1
AccountInfo= AccountInfo[['USERID','FirstloginRange']]

#Find the characteristics about transaction: Period, average transaction amount..
transactionInfo= transaction
transactionInfo['DATE2']=transactionInfo['DATE']
transactionInfo.head()
transactionInfo=(transactionInfo.groupby(['USERID'])
          .agg({ 'TOTAL':'sum','DATE':'min','DATE2':'max','UNITS':'count'}).reset_index())


transactionInfo.columns=['USERID','TOTALTransaction','Mindate','Maxdate','NumberofTrans']


transactionInfo['Period']=((transactionInfo['Maxdate']-transactionInfo['Mindate']).astype('timedelta64[D]')+1) /transactionInfo['NumberofTrans']
transactionInfo['AveTrans']=transactionInfo['TOTALTransaction']/transactionInfo['NumberofTrans']


transactionInfo=transactionInfo[['USERID','Period','AveTrans']]
transactionInfo.head()


UserInfo=pd.merge(AccountInfo,transactionInfo, on=['USERID'], how='left')
UserInfo=UserInfo.round(2) 
UserInfo=UserInfo.fillna(0)
UserInfo.head()

FirstLogginBefore20160403['Year'] = FirstLogginBefore20160403['DATE'].map(lambda x: x.year)
FirstLogginBefore20160403['Month'] = FirstLogginBefore20160403['DATE'].map(lambda x: x.month)

transaction['Year'] = transaction['DATE'].map(lambda x: x.year)
transaction['Month'] = transaction['DATE'].map(lambda x: x.month)


marketing['Year'] = marketing['DATE'].map(lambda x: x.year)
marketing['Month'] = marketing['DATE'].map(lambda x: x.month)

marketing['MARKETINGEVENT'].unique()


# In[19]:


Mar1=marketing.loc[marketing['MARKETINGEVENT']=='LAUNCH CAMPAIGN']
Mar2=marketing.loc[marketing['MARKETINGEVENT']=='MINOR EVENT SPONSORSHIP']
Mar3=marketing.loc[marketing['MARKETINGEVENT']=='MAJOR ADVERTISING BUY']
Mar4=marketing.loc[marketing['MARKETINGEVENT']=='MINOR ADVERTISING BUY']
Mar5=marketing.loc[marketing['MARKETINGEVENT']=='MAJOR EVENT SPONSORSHIP']


# In[20]:


Mar1=Mar1.groupby(['Year','Month'])['MARKETINGEVENT'].agg('count').reset_index()
Mar1.columns=['Year','Month','LAUNCH CAMPAIGN']
Mar2=Mar2.groupby(['Year','Month'])['MARKETINGEVENT'].agg('count').reset_index()
Mar2.columns=['Year','Month','MINOR EVENT SPONSORSHIP']
Mar3=Mar3.groupby(['Year','Month'])['MARKETINGEVENT'].agg('count').reset_index()
Mar3.columns=['Year','Month','MAJOR ADVERTISING BUY']
Mar4=Mar4.groupby(['Year','Month'])['MARKETINGEVENT'].agg('count').reset_index()
Mar4.columns=['Year','Month','MINOR ADVERTISING BUY']
Mar5=Mar5.groupby(['Year','Month'])['MARKETINGEVENT'].agg('count').reset_index()
Mar5.columns=['Year','Month','MAJOR EVENT SPONSORSHIP']


# In[21]:


Newmarketing=marketing[['Year','Month']]
Newmarketing=Newmarketing.drop_duplicates(subset=['Year','Month'], keep='first')


# In[22]:


TotalMar=pd.merge(Newmarketing,Mar1, on=['Year','Month'], how='left')
TotalMar=pd.merge(TotalMar,Mar2, on=['Year','Month'], how='left')
TotalMar=pd.merge(TotalMar,Mar3, on=['Year','Month'], how='left')
TotalMar=pd.merge(TotalMar,Mar4, on=['Year','Month'], how='left')
TotalMar=pd.merge(TotalMar,Mar5, on=['Year','Month'], how='left')
TotalMar=TotalMar.fillna(0)


# In[23]:


pricing['Year'] = pricing['DATE'].map(lambda x: x.year)
pricing['Month'] = pricing['DATE'].map(lambda x: x.month)


# In[24]:


TotalPrice=pricing.groupby(['Year','Month'])['PRICE'].mean().reset_index()


# In[25]:


TotalPriceandMar=pd.merge(TotalMar,TotalPrice, on=['Year','Month'], how='left')
TotalPriceandMar.head()


# In[26]:


TransactionSummary=transaction.groupby(['Year','Month','USERID'])['TOTAL'].agg('sum').reset_index()


# In[27]:


TransactionSummary=pd.merge(TransactionSummary,TotalPriceandMar, on=['Year','Month'], how='left')


# In[28]:


TransactionSummary.head()


# In[29]:


#FirstLogginBefore20160403.head()
AllTransactions=pd.merge(FirstLogginBefore20160403,TransactionSummary, on=['USERID'], how='left')


# In[30]:


AllTransactions.head()


# In[31]:


AllTransactions['LoginMonthCal']=AllTransactions['Year_x']*12+AllTransactions['Month_x']
AllTransactions['TransMonthCal']=AllTransactions['Year_y']*12+AllTransactions['Month_y']


# In[32]:


#Just use trans data of next 3 months
AllTransactions=AllTransactions.loc[AllTransactions['LoginMonthCal']<AllTransactions['TransMonthCal']]
AllTransactions=AllTransactions.loc[AllTransactions['LoginMonthCal']>(AllTransactions['TransMonthCal']-4)]


# In[33]:


AllTransactions.head(5)


# In[34]:


#TotalTransaction=AllTransactions.groupby(['USERID','DATE'])['TOTAL'].agg('sum').reset_index()
TotalTransaction=(AllTransactions.groupby(['USERID','DATE'])
          .agg({'TOTAL':'sum', 'LAUNCH CAMPAIGN':'sum','MINOR EVENT SPONSORSHIP':'sum','MAJOR ADVERTISING BUY':'sum',
               'MINOR ADVERTISING BUY':'sum','MAJOR EVENT SPONSORSHIP':'sum','PRICE':'mean'}).reset_index())


# In[35]:


TotalTransaction.rename(columns={'DATE': 'Firstlogin'}, inplace=True)
TotalTransaction['Highvalue'] = np.where(TotalTransaction['TOTAL']>=100, 1, 0)
Labeldata=TotalTransaction


# In[36]:


Labeldata=pd.merge(Labeldata,UserInfo, on=['USERID'], how='left')


# In[37]:


Labeldata.head()


# In[38]:


Labeldata.groupby('Highvalue').count()


# In[40]:


Newaccount=pd.read_csv('data/problem-two-new-users.csv',header=None)


# In[41]:


Checknewaccount=Newaccount
Checknewaccount.columns=['USERID']


# In[42]:


Checknewaccount=pd.merge(Checknewaccount,Labeldata, on=['USERID'], how='left')
Checknewaccount2=Checknewaccount.fillna(0)
Checknewaccount2.head()


# In[43]:


DontPredict=Checknewaccount2.loc[Checknewaccount2['Firstlogin'] == 0]
NeedPredict=Checknewaccount2.loc[Checknewaccount2['Firstlogin'] != 0]


# In[44]:


Result_DontPredict=DontPredict[['USERID']]
Result_DontPredict['result']=0


# In[45]:


TotalPriceandMarPredict=TotalPriceandMar.loc[(TotalPriceandMar['Year'] == 2016)&(TotalPriceandMar['Month'] > 5)]


# In[46]:


TotalPriceandMarPredict=(TotalPriceandMarPredict.groupby(['Year'])
          .agg({ 'LAUNCH CAMPAIGN':'sum','MINOR EVENT SPONSORSHIP':'sum','MAJOR ADVERTISING BUY':'sum',
               'MINOR ADVERTISING BUY':'sum','MAJOR EVENT SPONSORSHIP':'sum','PRICE':'mean'}).reset_index())


# In[47]:


TotalPriceandMarPredict


# In[48]:


NeedPredict['MINOR EVENT SPONSORSHIP']=2
NeedPredict['MAJOR ADVERTISING BUY']=1
NeedPredict['MINOR ADVERTISING BUY']=1
NeedPredict['MAJOR EVENT SPONSORSHIP']=1
NeedPredict['PRICE']=3.779928


# In[49]:


Labeldata=Labeldata.loc[Labeldata['Firstlogin']<'2016-04-01']
#Labeldata=Labeldata.loc[Labeldata['Firstlogin']>='2016-01-01']


# In[50]:


Labeldata.groupby('Highvalue').count()


# In[51]:


Checknewaccount3=Newaccount
Checknewaccount3['CHECK']='Y'
Checknewaccount3.drop_duplicates(keep='first').reset_index()
Checknewaccount3.head()


# In[52]:


Labeldata=Labeldata.merge(Checknewaccount3, on=['USERID'], how='left')
Labeldata=Labeldata.loc[Labeldata['CHECK']!='Y']


# In[53]:


Labeldata.groupby('Highvalue').count()


# In[54]:


Labeldata=Labeldata.round(2) 
Labeldata=Labeldata.fillna(0)
Labeldata.head()


# In[55]:


#X = Labeldata[['LAUNCH CAMPAIGN','MINOR EVENT SPONSORSHIP','MAJOR ADVERTISING BUY',
#              'MINOR ADVERTISING BUY','MAJOR EVENT SPONSORSHIP','PRICE']]
X = Labeldata[['PRICE','FirstloginRange','Period','AveTrans','LAUNCH CAMPAIGN','MINOR EVENT SPONSORSHIP',
               'MAJOR ADVERTISING BUY', 'MINOR ADVERTISING BUY','MAJOR EVENT SPONSORSHIP']]
y = Labeldata['Highvalue']


# In[56]:


print("Training step:")
print("Call the model and split the data")

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
#sm = SMOTE(random_state=42)
#X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)
# instantiate the model (using the default parameters)
lr_model_single = LogisticRegression()
X_test=X_test.fillna(0)


# In[70]:


print("fitting model with data")
lr_model_single.fit(X_train,y_train)


# In[73]:


print ("Serializing metadata.....")
dump(lr_model_single, 'CustomerClassification.joblib')



