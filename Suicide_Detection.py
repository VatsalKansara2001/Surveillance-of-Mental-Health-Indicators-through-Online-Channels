import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Read data
df = pd.read_csv(r"C:\DJ\Mtech - Data Science\Activity_Cmpetition\Datathon 2.0\Dataset\Suicide_Ideation_Dataset(Twitter-based).csv")


# Handling missing values
df.dropna(inplace=True)

# Encode categorical labels into numeric form
le = LabelEncoder()
df['Suicide'] = le.fit_transform(df['Suicide'])

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=2000)
X = vectorizer.fit_transform(df['Tweet']).toarray()
y = df['Suicide']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Building a neural network model using TensorFlow
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=32)

# Training various machine learning models
models = [
    ('XGBClassifier', XGBClassifier()),
    ('CatBoostClassifier', CatBoostClassifier(verbose=0)),
    ('LGBMClassifier', LGBMClassifier()),
    ('RandomForestClassifier', RandomForestClassifier()),
    ('SVC', SVC(probability=True))
]

model_scores = {}  # To store accuracy scores

for model_name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_pred, y_test)
    model_scores[model_name] = score
    print(f"{model_name} Accuracy Score: {score:.2f}")

# Ensemble learning using Voting Classifier
model_instances = [model for _, model in models]
voting_classifier = VotingClassifier(estimators=models, voting='soft')
voting_classifier.fit(X_train, y_train)
y_pred2 = voting_classifier.predict(X_test)
voting_score = accuracy_score(y_pred2, y_test)
model_scores["Voting Ensemble"] = voting_score
print(f"Voting Ensemble Accuracy Score: {voting_score:.2f}")

# Plotting the accuracy scores
plt.figure(figsize=(10, 6))
plt.bar(model_scores.keys(), model_scores.values(), color='skyblue')
plt.xlabel('Models')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores of Different Models')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1)
plt.show()
