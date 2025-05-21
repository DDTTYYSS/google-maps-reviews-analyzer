import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime

class ReviewDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

def analyze_sentiment(reviews_df, batch_size=16):
    # Load pre-trained BERT model and tokenizer
    model_name = 'bert-base-chinese'  # Using Chinese BERT for better Chinese text understanding
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)  # 3 labels: negative, neutral, positive
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Create dataset and dataloader
    dataset = ReviewDataset(reviews_df['review_text'].values, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Initialize lists to store predictions
    predictions = []
    
    # Process reviews in batches
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Analyzing reviews"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            predictions.extend(preds.cpu().numpy())

    # Convert predictions to sentiment labels
    sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    sentiments = [sentiment_map[pred] for pred in predictions]
    
    # Add sentiment analysis results to dataframe
    reviews_df['sentiment'] = sentiments
    
    return reviews_df

def main():
    # Get the most recent CSV file from data_store directory
    data_dir = "data_store"
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not csv_files:
        print("No CSV files found in data_store directory")
        return
    
    latest_file = max(csv_files, key=lambda x: os.path.getctime(os.path.join(data_dir, x)))
    file_path = os.path.join(data_dir, latest_file)
    
    print(f"Analyzing reviews from: {file_path}")
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Analyze sentiment
    df_with_sentiment = analyze_sentiment(df)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(data_dir, f"analyzed_{latest_file}")
    df_with_sentiment.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    # Print summary
    print("\nSentiment Analysis Summary:")
    sentiment_counts = df_with_sentiment['sentiment'].value_counts()
    for sentiment, count in sentiment_counts.items():
        print(f"{sentiment}: {count} reviews ({count/len(df_with_sentiment)*100:.1f}%)")
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main() 