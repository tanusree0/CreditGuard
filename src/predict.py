import joblib
import numpy as np

model=joblib.load(
    "artifacts/model.pkl"
)

scaler=joblib.load(
    "artifacts/scaler.pkl"
)

age=int(input("Age: "))
income=float(input("Income: "))
loan=float(input("Loan Amount: "))
score=int(input("Credit Score: "))
years=int(input("Employment Years: "))

data=np.array([
    [
        age,
        income,
        loan,
        score,
        years
    ]
])

data=scaler.transform(data)

prediction=model.predict(data)[0]

probability=max(
    model.predict_proba(data)[0]
)

if prediction==0:
    print("Safe Customer")
else:
    print("Risky Customer")

print(
    f"Confidence: {probability*100:.2f}%"
)