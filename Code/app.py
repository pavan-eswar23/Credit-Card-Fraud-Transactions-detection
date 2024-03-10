import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st


dt=pd.read_csv('creditcard.csv')
normal_tr=dt[dt['Class']==0]
fraud_tr=dt[dt['Class']==1]

normal_sample=normal_tr.sample(492)
newdt=pd.concat([normal_sample,fraud_tr],axis=0)

x=newdt.drop('Class',axis=1)
y=newdt['Class']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=42)
logr=LogisticRegression()
logr.fit(x_train,y_train)

train_acc = accuracy_score(logr.predict(x_train), y_train)
test_acc = accuracy_score(logr.predict(x_test), y_test)

# create Streamlit app
st.title("Credit Card Fraud Detection Model")
st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")

input_df = st.text_input('Input All features')
input_df_lst = input_df.split(',')
submit = st.button("Submit")

if submit:
    # get input feature values
    features = np.array(input_df_lst, dtype=np.float64)
    # make prediction
    prediction = logr.predict(features.reshape(1,-1))
    # display result
    if prediction[0] == 0:
        st.write("Legitimate transaction")
    else:
        st.write("Fraudulent transaction")