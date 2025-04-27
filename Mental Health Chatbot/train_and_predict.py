import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

dataset_path = 'C:/Users/ethan/Desktop/Uni Work/Year 3/Honours Project/datasets'

if not os.path.exists(dataset_path):
    print(f"Dataset folder '{dataset_path}' does not exist.")
    exit(1)
else:
    print(f"Reading datasets from existing folder: '{dataset_path}'")

# Load all CSVs in the folder and assign labels based on filename
dataframes = []
for filename in os.listdir(dataset_path):
    if filename.endswith('.csv') and filename != '.DS_Store':
        label = filename.split('_')[0].lower()
        file_path = os.path.join(dataset_path, filename)

        df = pd.read_csv(file_path)

        if 'text' not in df.columns:
            print(f"Skipping {filename}: no 'text' column.")
            continue

        df = df[['text']].copy()
        df['label'] = label
        dataframes.append(df)

# 3. Combine all datasets
full_df = pd.concat(dataframes, ignore_index=True)
print(f"Loaded {len(full_df)} total text samples from {len(dataframes)} categories.")

# 4. Vectorize the text
texts = full_df['text']
labels = full_df['label']

vectorizer = TfidfVectorizer(max_features=512)
X = vectorizer.fit_transform(texts)
y = labels

# 5. Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 6. Save the model and vectorizer
joblib.dump(model, 'mental_health_multi_model.pkl')
joblib.dump(vectorizer, 'tfidf_multi_vectorizer.pkl')

print("Model training complete.")
print("Saved 'mental_health_multi_model.pkl' and 'tfidf_multi_vectorizer.pkl'")
