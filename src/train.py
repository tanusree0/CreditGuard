import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/credit.csv")

X = df.drop("Default", axis=1)
y = df["Default"]

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train,y_train)

pred=model.predict(X_test)

acc=accuracy_score(
    y_test,
    pred
)

print("Accuracy:",acc)

joblib.dump(
    model,
    "artifacts/model.pkl"
)

joblib.dump(
    scaler,
    "artifacts/scaler.pkl"
)

print("Model Saved")