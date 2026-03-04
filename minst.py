import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn.ensemble import RandomForestClassifier


minst=fetch_openml('mnist_784',version=1)

x,y=minst.data,minst.target.astype(int)
print(x.shape)
print(y.shape)


fig,axis=plt.subplots(2,5,figsize=(8,4))
for i, ax in enumerate(axis.ravel()):
    ax.imshow(x.iloc[i].values.reshape(28,28),cmap='gray')
    ax.set_title(f"Label: {y[i]}")
    ax.axis('off')
    
plt.tight_layout()
# plt.show()

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

model=RandomForestClassifier(n_estimators=200,n_jobs=100)
model.fit(x_train_scaled,y_train)

print("Train score:", model.score(x_train_scaled, y_train))
print("Test score:", model.score(x_test_scaled, y_test))

y_pred = model.predict(x_test_scaled)


print("Precision Score:", precision_score(y_test, y_pred, average='macro'))
print("Recall Score:", recall_score(y_test, y_pred, average='macro'))
print("F1 Score:", f1_score(y_test, y_pred, average='macro'))
