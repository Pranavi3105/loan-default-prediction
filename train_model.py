import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import joblib

df = pd.read_csv("loan_data.csv")
features = [
    "credit.policy",
    "purpose",
    "int.rate",
    "installment",
    "log.annual.inc",
    "dti",
    "fico",
    "days.with.cr.line",
    "revol.bal",
    "revol.util",
    "inq.last.6mths",
    "delinq.2yrs",
    "pub.rec"
]
target = "not.fully.paid"
df = df[features + [target]]
label = LabelEncoder()
df["purpose"] = label.fit_transform(df["purpose"])
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
model = LGBMClassifier()
model.fit(X_train_smote, y_train_smote)
joblib.dump(model, "model.pkl")
joblib.dump(features, "feature_names.pkl")
print("Model saved successfully!")