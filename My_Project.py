#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings 
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.express as px


# In[2]:


df=pd.read_csv("heart_failure_clinical_records_dataset.csv")


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


df.isnull().sum()


# In[6]:


labels = ['No Diabetes','Diabetes']
diabetes_yes = df[df['diabetes']==1]
diabetes_no = df[df['diabetes']==0]
values = [len(diabetes_no), len(diabetes_yes)]
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5)])
fig.update_layout(
    title_text="Analysis on Diabetes")
fig.show()


# In[7]:


fig=px.pie(df,values='diabetes',names='DEATH_EVENT',title='Death Analysis')
fig.show()


# In[42]:


plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),vmin=-1,cmap='coolwarm',annot=True);


# In[43]:


ax=sns.barplot(x='high_blood_pressure',y='platelets',hue='DEATH_EVENT',data=df)
plt.legend(loc=9)
plt.show()


# In[44]:


sns.distplot(df['creatinine_phosphokinase'], kde= True)
plt.show()


# In[45]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.kdeplot(df['creatinine_phosphokinase'])

plt.subplot(1,2,2)
sns.boxplot(df['creatinine_phosphokinase'])
plt.show()


# ## LogisticRegression

# In[46]:


from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import confusion_matrix,accuracy_score


# In[9]:


Feature=['time','ejection_fraction','serum_creatinine']
x=df[Feature]
y=df["DEATH_EVENT"]


# In[48]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)


# In[49]:


from sklearn.linear_model import LogisticRegression


# In[50]:


log_re=LogisticRegression()


# In[51]:


log_re.fit(x_train,y_train)
log_re_pred=log_re.predict(x_test)


# In[52]:


log_acc=accuracy_score(y_test,log_re_pred)
print("Logistic Accuracy Score: ","{:.2f}%".format(100*log_acc))


# ## Decision tree

# In[53]:


import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("heart_failure_clinical_records_dataset.csv")

t = np.array(list(df['creatinine_phosphokinase'])).reshape(-1, 1)
pt = PowerTransformer(method = "yeo-johnson")
creatinine_phosphokinase = pt.fit_transform(t)
df['creatinine_phosphokinase'] = creatinine_phosphokinase

t = np.array(list(df['serum_creatinine'])).reshape(-1, 1)
pt = PowerTransformer(method = "yeo-johnson")
serum_creatinine = pt.fit_transform(t)
df['serum_creatinine'] = serum_creatinine

df.drop(columns = ['sex', 'diabetes'], inplace = True)
X = df.iloc[:, 0:10].values
Y = df['DEATH_EVENT'].values

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=6)

dclf = DecisionTreeClassifier(criterion='gini',
                              max_depth=15,
                            )
dclf.fit(x_train, y_train)

pickle.dump(dclf, open('dclf.pkl', 'wb'))

clf = pickle.load(open('dclf.pkl', 'rb'))
print("Decision Tree :",clf.score(x_test, y_test))


# ## XGBRF Classifier

# In[54]:


import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer
from xgboost import XGBRFClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("heart_failure_clinical_records_dataset.csv")

t = np.array(list(df['creatinine_phosphokinase'])).reshape(-1, 1)
pt = PowerTransformer(method = "yeo-johnson")
creatinine_phosphokinase = pt.fit_transform(t)
df['creatinine_phosphokinase'] = creatinine_phosphokinase

t = np.array(list(df['serum_creatinine'])).reshape(-1, 1)
pt = PowerTransformer(method = "yeo-johnson")
serum_creatinine = pt.fit_transform(t)
df['serum_creatinine'] = serum_creatinine

df.drop(columns = ['sex', 'diabetes'], inplace = True)
X = df.iloc[:, 0:10].values
Y = df['DEATH_EVENT'].values

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=6)

xrclf = XGBRFClassifier()
xrclf.fit(x_train, y_train)
y_pred=xrclf.predict(x_test)

log_acc=accuracy_score(y_test,y_pred)
print("XGBRF Accuracy Score: ","{:.2f}%".format(100*log_acc))


# ## XGBoost Classifier

# In[55]:


import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("heart_failure_clinical_records_dataset.csv")

t = np.array(list(df['creatinine_phosphokinase'])).reshape(-1, 1)
pt = PowerTransformer(method = "yeo-johnson")
creatinine_phosphokinase = pt.fit_transform(t)
df['creatinine_phosphokinase'] = creatinine_phosphokinase

t = np.array(list(df['serum_creatinine'])).reshape(-1, 1)
pt = PowerTransformer(method = "yeo-johnson")
serum_creatinine = pt.fit_transform(t)
df['serum_creatinine'] = serum_creatinine

df.drop(columns = ['sex', 'diabetes'], inplace = True)
X = df.iloc[:, 0:10].values
Y = df['DEATH_EVENT'].values

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=6)

xclf = XGBClassifier()
xclf.fit(x_train, y_train)

pickle.dump(xclf, open('xclf.pkl', 'wb'))

clf = pickle.load(open('xclf.pkl', 'rb'))
print("XGBoost Classifier:",clf.score(x_test, y_test))


# ## KNeighborsClassfier

# In[56]:


from sklearn.metrics import classification_report 
from sklearn.neighbors import KNeighborsClassifier

model2 = KNeighborsClassifier() 
model2.fit(x_train, y_train)

y_pred2 = model2.predict(x_test) 
print(classification_report(y_test, y_pred2))


# ## SVM

# In[57]:


from sklearn.metrics import classification_report 
from sklearn.svm import SVC

model3 = SVC(random_state=1) 
model3.fit(x_train, y_train) 

y_pred3 = model3.predict(x_test) 
print(classification_report(y_test, y_pred3))


# ## LogisticRegression

# In[58]:


from sklearn.metrics import classification_report 
from sklearn.linear_model import LogisticRegression

model1 = LogisticRegression(random_state=1) 
model1.fit(x_train, y_train)  

y_pred1 = model1.predict(x_test) 
print(classification_report(y_test, y_pred1))


# ## DecisionTree

# In[59]:


from sklearn.metrics import classification_report 
from sklearn.tree import DecisionTreeClassifier

model5 = DecisionTreeClassifier(random_state=1) 
model5.fit(x_train, y_train) 

y_pred5 = model5.predict(x_test) 
print(classification_report(y_test, y_pred5))


# ## XGBoostClassifier

# In[60]:


from xgboost import XGBClassifier

model7 = XGBClassifier(random_state=1)
model7.fit(x_train, y_train)
y_pred7 = model7.predict(x_test)
print(classification_report(y_test, y_pred7))


# ## RandomForestClassifier

# In[61]:


from sklearn.metrics import classification_report 
from sklearn.ensemble import RandomForestClassifier

model6 = RandomForestClassifier(random_state=1)
model6.fit(x_train, y_train)  

y_pred6 = model6.predict(x_test) 
print(classification_report(y_test, y_pred6))


# In[63]:


importance = model6.feature_importances_

for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))


# In[83]:


index= df.columns[:-1]
importance = pd.Series(model6.feature_importances_, index=index)
importance.nlargest(13).plot(kind='barh', colormap='winter')


# In[ ]:




