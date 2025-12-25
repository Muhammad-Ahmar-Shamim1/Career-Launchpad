import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)

    # Column structure: ID, Title, Sentiment, Text
    df.columns = ["id", "title", "sentiment", "text"]

    df["clean_text"] = df["text"].apply(clean_text)
    df["label"] = df["sentiment"].apply(lambda x: 1 if x == "Positive" else 0)

    return df[["clean_text", "label"]]
