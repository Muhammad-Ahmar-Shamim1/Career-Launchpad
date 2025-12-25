import os
import torch
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from src.preprocess import clean_text

# Set page config
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_DIR = os.path.join(PROJECT_DIR, "models", "bert_sentiment_model")

# CSS styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .sentiment-positive {
        color: #28a745;
        font-weight: bold;
        font-size: 24px;
    }
    .sentiment-negative {
        color: #dc3545;
        font-weight: bold;
        font-size: 24px;
    }
    .confidence-high {
        color: #28a745;
    }
    .confidence-medium {
        color: #ffc107;
    }
    .confidence-low {
        color: #dc3545;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained BERT model and tokenizer"""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
        tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
        model.to(device)
        model.eval()
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def predict_sentiment(text, model, tokenizer, device):
    """Predict sentiment for the given text"""
    try:
        # Preprocess text
        cleaned_text = clean_text(text)
        
        # Tokenize
        inputs = tokenizer(
            cleaned_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
        
        probs = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = torch.max(probs).item()
        
        sentiment = "Positive" if prediction == 1 else "Negative"
        
        return sentiment, confidence, cleaned_text
    except Exception as e:
        return None, None, str(e)

# Main title
st.title("🎯 Sentiment Analysis with BERT")
st.markdown("---")

# Load model
with st.spinner("Loading BERT model..."):
    model, tokenizer, device = load_model()

if model is None:
    st.error("❌ Failed to load the model. Please ensure the model is trained first.")
    st.info("Run: python -m src.train_model")
else:
    st.success(f"✅ Model loaded successfully on {device}")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📁 Batch Analysis", "📈 Model Info"])
    
    # Tab 1: Single Prediction
    with tab1:
        st.header("Single Text Analysis")
        
        # Text input area
        user_input = st.text_area(
            "Enter text to analyze:",
            placeholder="Type or paste your text here...",
            height=150
        )
        
        # Prediction button
        col1, col2 = st.columns([1, 4])
        
        with col1:
            predict_button = st.button("🚀 Analyze", use_container_width=True)
        
        if predict_button and user_input:
            with st.spinner("Analyzing..."):
                sentiment, confidence, cleaned = predict_sentiment(
                    user_input, model, tokenizer, device
                )
            
            if sentiment is not None:
                # Display results
                st.markdown("---")
                st.subheader("📊 Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Sentiment",
                        sentiment,
                        delta="Positive ✅" if sentiment == "Positive" else "Negative ❌"
                    )
                
                with col2:
                    confidence_pct = confidence * 100
                    st.metric(
                        "Confidence",
                        f"{confidence_pct:.2f}%"
                    )
                
                with col3:
                    confidence_class = "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "low"
                    confidence_text = "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"
                    st.metric("Confidence Level", confidence_text)
                
                # Display cleaned text
                with st.expander("📝 Cleaned Text (for reference)"):
                    st.text(cleaned)
                
                # Confidence visualization
                st.markdown("---")
                st.subheader("📈 Confidence Distribution")
                
                confidence_data = {
                    "Negative": 1 - confidence if sentiment == "Positive" else confidence,
                    "Positive": confidence if sentiment == "Positive" else 1 - confidence
                }
                
                chart_data = pd.DataFrame({
                    "Sentiment": list(confidence_data.keys()),
                    "Confidence": list(confidence_data.values())
                })
                
                st.bar_chart(chart_data.set_index("Sentiment"))
            
            else:
                st.error(f"Error during prediction: {cleaned}")
        
        elif predict_button and not user_input:
            st.warning("Please enter text to analyze!")
    
    # Tab 2: Batch Analysis
    with tab2:
        st.header("Batch Text Analysis")
        
        uploaded_file = st.file_uploader(
            "Upload a CSV file with texts",
            type="csv",
            help="CSV should have a 'text' column"
        )
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            if "text" not in df.columns:
                st.error("CSV must contain a 'text' column")
            else:
                st.info(f"Loaded {len(df)} rows")
                
                if st.button("🚀 Analyze All"):
                    with st.spinner("Analyzing texts..."):
                        results = []
                        progress_bar = st.progress(0)
                        
                        for idx, row in df.iterrows():
                            sentiment, confidence, _ = predict_sentiment(
                                row["text"], model, tokenizer, device
                            )
                            results.append({
                                "text": row["text"][:50] + "..." if len(row["text"]) > 50 else row["text"],
                                "sentiment": sentiment,
                                "confidence": confidence
                            })
                            progress_bar.progress((idx + 1) / len(df))
                    
                    results_df = pd.DataFrame(results)
                    
                    # Display statistics
                    col1, col2, col3 = st.columns(3)
                    
                    positive_count = (results_df["sentiment"] == "Positive").sum()
                    negative_count = (results_df["sentiment"] == "Negative").sum()
                    avg_confidence = results_df["confidence"].mean()
                    
                    with col1:
                        st.metric("Positive", positive_count)
                    with col2:
                        st.metric("Negative", negative_count)
                    with col3:
                        st.metric("Avg Confidence", f"{avg_confidence:.2%}")
                    
                    # Display table
                    st.markdown("---")
                    st.subheader("📋 Results Table")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Download Results as CSV",
                        data=csv,
                        file_name="sentiment_results.csv",
                        mime="text/csv"
                    )
    
    # Tab 3: Model Info
    with tab3:
        st.header("Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔧 Model Details")
            st.markdown(f"""
            - **Model Type**: BERT (Bidirectional Encoder Representations from Transformers)
            - **Model Name**: bert-base-uncased
            - **Number of Labels**: 2 (Positive, Negative)
            - **Device**: {device}
            - **Model Location**: {MODEL_DIR}
            """)
        
        with col2:
            st.subheader("⚙️ Configuration")
            st.markdown("""
            - **Max Token Length**: 128
            - **Batch Size**: 8
            - **Training Epochs**: 2
            - **Learning Rate**: 2e-5
            - **Optimizer**: AdamW
            """)
        
        st.markdown("---")
        st.subheader("📖 How It Works")
        st.markdown("""
        1. **Input**: User provides text for sentiment analysis
        2. **Preprocessing**: Text is cleaned (lowercase, remove URLs, stopwords, etc.)
        3. **Tokenization**: Text is tokenized using BERT tokenizer
        4. **Model**: BERT processes tokens and outputs sentiment logits
        5. **Output**: Softmax converts logits to confidence scores
        6. **Result**: Sentiment (Positive/Negative) with confidence percentage
        """)
        
        st.markdown("---")
        st.subheader("💡 Tips")
        st.markdown("""
        - Longer texts work better (minimum 20 characters recommended)
        - The model works best for review-like content
        - Confidence > 80% is considered high confidence
        - For batch processing, use a CSV with a 'text' column
        """)
