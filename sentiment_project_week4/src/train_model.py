import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.preprocess import load_and_preprocess

MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 8
EPOCHS = 2

# Set paths relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_DIR, "data", "reviews.csv")
MODEL_OUTPUT_DIR = os.path.join(PROJECT_DIR, "models", "bert_sentiment_model")

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True, max_length=128
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

try:
    # Load data
    print(f"Loading data from: {DATA_PATH}")
    df = load_and_preprocess(DATA_PATH)
    print(f"Data loaded successfully. Records: {len(df)}")

    X_train, X_val, y_train, y_val = train_test_split(
        df['clean_text'], df['label'], test_size=0.2, random_state=42
    )

    train_dataset = SentimentDataset(X_train.tolist(), y_train.tolist())
    val_dataset = SentimentDataset(X_val.tolist(), y_val.tolist())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # Training
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} Training Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        total_val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                total_val_loss += loss.item()

                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1)
                val_preds.extend(predictions.cpu().numpy())
                val_labels.extend(batch["labels"].cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        accuracy = accuracy_score(val_labels, val_preds)
        precision = precision_score(val_labels, val_preds, zero_division=0)
        recall = recall_score(val_labels, val_preds, zero_division=0)
        f1 = f1_score(val_labels, val_preds, zero_division=0)

        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # Create output directory if it doesn't exist
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    # Save model
    print(f"Saving model to: {MODEL_OUTPUT_DIR}")
    model.save_pretrained(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    print("Model saved successfully!")

except FileNotFoundError as e:
    print(f"Error: Data file not found - {e}")
except Exception as e:
    print(f"Error during training: {e}")
