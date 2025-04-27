import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

dataset_path = 'C:/Users/ethan/Desktop/Uni Work/Year 3/Honours Project/datasets'
model_output_path = 'mental_health_multi_model.pkl'
scaler_output_path = 'mental_health_scaler.pkl'

if not os.path.exists(dataset_path):
    print(f"Dataset folder '{dataset_path}' does not exist.")
    exit(1)
else:
    print(f"Reading datasets from existing folder: '{dataset_path}'")

#Load all CSVs in the folder and assign labels based on filename
dataframes = []
for filename in os.listdir(dataset_path):
    if filename.endswith('.csv'):
        label = filename.split('_')[0].lower()
        file_path = os.path.join(dataset_path, filename)

        df = pd.read_csv(file_path)

        #Drop non-numeric columns (keep only features)
        df = df.drop(columns=['subreddit', 'author', 'date', 'post'], errors='ignore')

        df['label'] = label
        dataframes.append(df)

#Combine all datasets
full_df = pd.concat(dataframes, ignore_index=True)
print(f"Loaded {len(full_df)} total text samples from {len(dataframes)} categories.")

#Prepare X (features) and y (labels)
X = full_df.drop('label', axis=1)
y = full_df['label']

#Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Split and train
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=6000)
model.fit(X_train, y_train)

#Save the model and vectorizer
joblib.dump(model, 'mental_health_multi_model.pkl')
joblib.dump(scaler, scaler_output_path)

print("Model training complete.")
print("Saved 'mental_health_multi_model.pkl' and 'tfidf_multi_vectorizer.pkl'")
print(f"📦 Saved model to '{model_output_path}' and scaler to '{scaler_output_path}'")
