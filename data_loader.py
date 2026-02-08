# data_loader.py
import kagglehub
import pandas as pd
import os

path = kagglehub.dataset_download("adisongoh/it-service-ticket-classification-dataset")
print("Dataset path:", path)

# Load the exact file
df = pd.read_csv(os.path.join(path, "all_tickets_processed_improved_v3.csv"))
print("Shape:", df.shape)
print("\\nLabels:")
print(df['Topic_group'].value_counts())
df.to_csv("data/raw/tickets.csv", index=False)