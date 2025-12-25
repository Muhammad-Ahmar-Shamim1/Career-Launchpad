# Sentiment Analysis with BERT

A sentiment classification project using BERT (Bidirectional Encoder Representations from Transformers) to classify reviews as positive or negative.

## Project Structure

```
sentiment_project_week4/
├── data/
│   └── reviews.csv          # Training data (ID, Title, Sentiment, Text)
├── models/
│   └── bert_sentiment_model/ # Trained BERT model
├── notebooks/
│   ├── 01_data_loading_cleaning.ipynb
│   ├── 02_tokenization.ipynb
│   └── 03_bert_training.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocess.py        # Data preprocessing functions
│   ├── train_model.py       # Model training script
│   └── api.py               # Flask API for predictions
├── requirements.txt
└── README.md
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download required NLTK data (run once):
```python
import nltk
nltk.download('stopwords')
```

## Usage

### Training the Model

```bash
python -m src.train_model
```

This will:
- Load and preprocess the review data
- Split into training (80%) and validation (20%) sets
- Train BERT model for 2 epochs
- Validate and display metrics (Accuracy, Precision, Recall, F1)
- Save model to `models/bert_sentiment_model/`

### Using the API

1. Ensure the model is trained first
2. Start the API server:
```bash
python -m src.api
```

3. Make predictions via POST request:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is amazing!"}'
```

Response:
```json
{
  "sentiment": "Positive",
  "confidence": 0.98
}
```

4. Check API health:
```bash
curl http://localhost:5000/health
```

## Key Improvements Made

1. **Fixed Requirements** - Added all necessary dependencies with versions
2. **Path Handling** - Fixed relative path issues for cross-platform compatibility
3. **Validation Loop** - Added validation during training
4. **Metrics Tracking** - Added Accuracy, Precision, Recall, F1 scores
5. **Error Handling** - Added try-catch blocks for robustness
6. **Device Management** - Proper GPU/CPU detection and handling
7. **API Improvements** - Added error handling, health check endpoint, max_length for tokenizer
8. **Memory Optimization** - Added `torch.no_grad()` for inference to reduce memory usage

## Data Format

The `reviews.csv` file should have the following columns:
- ID: Review identifier
- Title: Review title
- Sentiment: "Positive" or "Negative"
- Text: Review text

## Configuration

Edit these variables in `src/train_model.py`:
- `BATCH_SIZE`: Training batch size (default: 8, reduce for low memory)
- `EPOCHS`: Number of training epochs (default: 2)
- `MODEL_NAME`: BERT model variant (default: "bert-base-uncased")

## Performance Optimization for Low-Memory Systems

1. Reduce `BATCH_SIZE` to 4 or 2
2. Reduce `max_length` from 128 to 64
3. Use `torch.cuda.empty_cache()` periodically
4. Consider using a quantized model variant

## Dependencies

- torch: Deep learning framework
- transformers: BERT model and tokenizer
- pandas: Data manipulation
- scikit-learn: Metrics and preprocessing
- nltk: Text preprocessing
- flask: Web API framework
