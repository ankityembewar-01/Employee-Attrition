#Basic imports
import numpy as np
import pandas as pd

#import machine learning model
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")

#reading data
df = pd.read_csv("Data.csv")

# label encoder for object type data
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
df[["OverTime","Attrition"]] = df[["OverTime","Attrition"]].apply(label.fit_transform)
df = np.array(df)

#defining feature & target variable
x = df[:,:-1] #feature
y = df[:,-1] #target variable
y = y.astype('float')
x = x.astype('float')

#spilting the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

#model initialization
model = ExtraTreesClassifier()

#model trainning
model.fit(X_train, y_train)



#b = log_reg.predict_proba(final)

# model dump using pickle
pickle.dump(model,open('model.pkl','wb'))
model1=pickle.load(open('model.pkl','rb'))

