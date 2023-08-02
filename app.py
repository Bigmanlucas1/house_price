#!/usr/bin/env python
# coding: utf-8

# In[1]:


# In[2]:


import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("heart_failure_clinical_records_dataset.csv")

t = np.array(list(df['creatinine_phosphokinase'])).reshape(-1, 1)
pt = PowerTransformer(method="yeo-johnson")
creatinine_phosphokinase = pt.fit_transform(t)
df['creatinine_phosphokinase'] = creatinine_phosphokinase

t = np.array(list(df['serum_creatinine'])).reshape(-1, 1)
pt = PowerTransformer(method="yeo-johnson")
serum_creatinine = pt.fit_transform(t)
df['serum_creatinine'] = serum_creatinine

df.drop(columns=['sex', 'diabetes'], inplace=True)
X = df.iloc[:, 0:10].values
Y = df['DEATH_EVENT'].values

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=6)

xclf = XGBClassifier()
xclf.fit(x_train, y_train)

pickle.dump(xclf, open('xclf.pkl', 'wb'))

clf = pickle.load(open('xclf.pkl', 'rb'))
print("XGBoost Classifier:", clf.score(x_test, y_test))


# In[3]:


from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('Pkl_Model.pkl', 'rb'))


# In[4]:




# In[9]:



@app.route('/')
@app.route('/index.html')
def home():
    return render_template("index.html")


# In[10]:




@app.route('/formdata.html')
def form():
    return render_template("formdata.html")

@app.route('/result.html')
def result():
    return render_template("result.html")


# In[11]:


@app.route('/about.html')
def about():
    return render_template("about.html")

@app.route('/predict', methods=['POST'])
def prediction():
    data1 = request.form['a']
    data2 = request.form['b']

    arr = np.array([[data1, data2]])
    # Predict the Model   
    pred = model.predict(arr)
    #print("Result: {0:.2f} %".format(100 * predict)) 
    return render_template('result.html', data=pred)


# In[ ]:


if __name__ == "__main__":
    app.run(debug=True)





