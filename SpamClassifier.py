import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ------------DATA PREPARATION-----------
df = pd.read_csv("Dataset/mail_data.csv")
data = df.where(pd.notnull(df), "")
data.loc[data["Category"] == "spam", "Category"] = 0
data.loc[data["Category"] == "ham", "Category"] = 1
X = data["Message"]
Y = data["Category"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
feature_extraction = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)

# ------------MODEL TRAINING------------
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)
Y_train = Y_train.astype("int")
Y_test = Y_test.astype("int")

model = LogisticRegression()
model.fit(X_train_features, Y_train)

# -----------TESTING MODEL-------------

accuarcyA = model.predict(X_train_features)
predictionA = accuracy_score(Y_train, accuarcyA)
accuarcyB = model.predict(X_test_features)
predictionB = accuracy_score(Y_test, accuarcyB)

email_body = ["""Congratulations! You've been selected for a chance to win a brand new iPhone 15 (even though it's not even released yet)!

We're giving away 10 lucky winners a chance to experience the future of smartphones absolutely FREE!  All you have to do is click the link below to claim your entry. But hurry, this offer won't last long!  ⏱️

Click Here to Claim Your FREE iPhone 15!  

P.S. Don't miss out on this incredible opportunity! Share this email with your friends and family for even more entries!

P.P.S.  We only have a limited number of phones available, so act fast!

Thanks,

The Totally Legit iPhone Giveaway Team"""]

email_body_features = feature_extraction.transform(email_body)
result = model.predict(email_body_features)

print("Accuracy of model on training data: ", predictionA)
print("Accuracy of model on testing data: ", predictionB)

print("---------------------------------------------------------------------")

if result == [0]:
    print("This email is a spam")

else:
    print("This email is not spam")
