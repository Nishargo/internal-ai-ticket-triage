import joblib
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

def train_pipeline():
    # Check if data exists, download if missing
    data_path = 'data/raw/tickets.csv'
    if not os.path.exists(data_path):
        print("Downloading dataset...")
        import kagglehub
        path = kagglehub.dataset_download("adisongoh/it-service-ticket-classification-dataset")
        
        # Find CSV
        csv_path = None
        for file in os.listdir(path):
            if file.endswith('.csv'):
                csv_path = os.path.join(path, file)
                break
        
        df = pd.read_csv(csv_path)
        os.makedirs('data/raw', exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"✅ Dataset saved: {df.shape}")
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"Loaded: {df.shape}")
    
    # Dataset: Document → Topic_group
    X = df['Document'].fillna('')  # ticket text
    y = df['Topic_group'].fillna('Other')  # ticket category
    
    print("Labels:", y.value_counts())
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # 80/20 split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )
    
    # TF-IDF pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),      # unigrams + bigrams
            max_features=5000,
            min_df=5,                # ignore rare terms
            max_df=0.95              # ignore too-common terms
        )),
        ('clf', LogisticRegression(
            random_state=42, 
            max_iter=1000,
            class_weight='balanced'  # handle imbalance
        ))
    ])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    print(f"\n🎯 Macro F1 Score: {f1_macro:.3f}")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Ensure models folder exists
    os.makedirs('models', exist_ok=True)
    
    # Save
    joblib.dump(pipeline, 'models/ticket_classifier.joblib')
    joblib.dump(le, 'models/label_encoder.joblib')
    
    print("\n✅ Models saved!")
    print(f"Classes: {le.classes_}")
    
    return pipeline, le

if __name__ == "__main__":
    train_pipeline()
