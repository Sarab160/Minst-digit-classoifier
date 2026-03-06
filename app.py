import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score,confusion_matrix,classification_report
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# 1 Load MNIST Dataset
# -----------------------------
mnist = fetch_openml('mnist_784', version=1)

X = mnist.data
y = mnist.target.astype(int)

print("Dataset Shape:", X.shape)
print("Labels Shape:", y.shape)

# -----------------------------
# 2 Show Sample Images
# -----------------------------
fig, axis = plt.subplots(2,5,figsize=(8,4))

for i, ax in enumerate(axis.ravel()):
    ax.imshow(X.iloc[i].values.reshape(28,28), cmap='gray')
    ax.set_title(f"Label: {y[i]}")
    ax.axis('off')

plt.tight_layout()
plt.show()
plt.savefig("sample digits..jpg")
# -----------------------------
# 3 Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4 Feature Scaling
# -----------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 5 Train Model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# -----------------------------
# 6 Model Performance
# -----------------------------
print("Train Score:", model.score(X_train_scaled, y_train))
print("Test Score:", model.score(X_test_scaled, y_test))

y_pred = model.predict(X_test_scaled)

print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall:", recall_score(y_test, y_pred, average='macro'))
print("F1 Score:", f1_score(y_test, y_pred, average='macro'))

conf=confusion_matrix(y_test,y_pred)


sns.heatmap(data=conf, annot=True, fmt='d', cmap='rocket')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
plt.savefig("confusion matrix.jpg")

# -----------------------------
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
# -----------------------------
# 7 Predict on Unseen Data
# -----------------------------

# take random unseen images from test set
random_samples = np.random.randint(0, len(X_test_scaled), 10)

unseen_images = X_test_scaled[random_samples]
true_labels = y_test.iloc[random_samples]

predictions = model.predict(unseen_images)

# -----------------------------
# 8 Visualize Predictions
# -----------------------------
fig, axes = plt.subplots(2,5, figsize=(10,4))

for i, ax in enumerate(axes.ravel()):
    
    image = unseen_images[i].reshape(28,28)
    
    ax.imshow(image, cmap='gray')
    ax.set_title(f"Pred: {predictions[i]}\nTrue: {true_labels.iloc[i]}")
    ax.axis('off')

plt.tight_layout()
plt.show()
plt.savefig("prediction.jpg")