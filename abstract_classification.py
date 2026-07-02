import ast
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from sentence_transformers import SentenceTransformer
import torch

#use our gpu
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

model = SentenceTransformer("google/embeddinggemma-300m")
model.to(device)


df = pd.read_csv("arxiv_data.csv")
df = df.dropna()

# terms are stored as Python list strings like ['cs.CV', 'cs.LG'], so parse them properly
df["terms"] = df["terms"].apply(ast.literal_eval)


subset = df.sample(n=20000, random_state=42)
train_subset, test_subset = train_test_split(subset, test_size=0.2, random_state=42)

# fit_transform on train labels, transform (no refit) on test labels
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(train_subset["terms"])
y_test = mlb.transform(test_subset["terms"])



#embed the text
print("Encoding train summaries...")
X_train = model.encode(train_subset["summaries"].tolist())
X_test = model.encode(test_subset["summaries"].tolist())



# OneVsRestClassifier trains one binary classifier per category
classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
classifier.fit(X_train, y_train)

test_predictions = classifier.predict(X_test)

# f1_score (samples) is more meaningful than accuracy for multi-label problems
test_f1 = f1_score(y_test, test_predictions, average="samples", zero_division=0)
accuracy = accuracy_score(y_test, test_predictions)
print(f"Test F1 (samples): {test_f1:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# show what categories were predicted for the first 5 samples
print("Sample predictions:", mlb.inverse_transform(test_predictions[:5]))



