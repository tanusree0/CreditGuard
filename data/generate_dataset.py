import pandas as pd
import numpy as np

np.random.seed(42)

rows = 50000

age = np.random.randint(18,65,rows)
income = np.random.randint(15000,200000,rows)
loan_amount = np.random.randint(5000,100000,rows)
credit_score = np.random.randint(300,850,rows)
employment_years = np.random.randint(0,40,rows)

default=[]

for i in range(rows):

    risk=0

    if credit_score[i] < 550:
        risk += 3

    if income[i] < 30000:
        risk += 2

    if loan_amount[i] > income[i]:
        risk += 2

    if employment_years[i] < 2:
        risk += 1

    if age[i] < 21:
        risk += 1

    prob=min(risk/10,0.95)

    label=np.random.choice(
        [0,1],
        p=[1-prob,prob]
    )

    default.append(label)

df=pd.DataFrame({
    "Age":age,
    "Income":income,
    "LoanAmount":loan_amount,
    "CreditScore":credit_score,
    "EmploymentYears":employment_years,
    "Default":default
})

df.to_csv("credit.csv",index=False)

print(df.head())
print(df.shape)