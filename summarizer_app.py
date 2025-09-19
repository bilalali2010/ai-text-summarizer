import streamlit as st
from transformers import pipeline
import torch

# Page configuration
st.set_page_config(
    page_title="Professional Text Summarizer",
    page_icon="✨",
    layout="wide"
)

# Custom CSS to improve the appearance
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    .summary-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown('<p class="main-header">✨ Advanced AI Summarizer</p>', unsafe_allow_html=True)
st.markdown("""
This tool uses Google's **Pegasus** model, specifically designed for high-quality summarization of news articles and long texts.
Paste your text below and get a concise, powerful summary in seconds.
""")

# Cache the model loading to avoid reloading on every interaction
@st.cache_resource(show_spinner=False)
def load_summarizer():
    """Load the summarization pipeline with the efficient Pegasus model."""
    # Use a smaller, faster model that is excellent for summarization
    model_name = "google/pegasus-cnn_dailymail"
    
    # Load the pipeline with specific settings to optimize for free tier
    summarizer = pipeline(
        "summarization",
        model=model_name,
        tokenizer=model_name,
        framework="pt",  # Use PyTorch
        device=0 if torch.cuda.is_available() else -1,  # Use GPU if available, else CPU
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32  # Use less memory on GPU
    )
    return summarizer

def main():
    # Initialize the summarizer
    with st.spinner("🧠 Loading the advanced AI model... This might take a minute for the first time."):
        summarizer = load_summarizer()
    
    st.success("Model loaded successfully!")
    
    # Create two columns for input and output
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Text")
        input_text = st.text_area(
            "Paste your article or long text here:",
            height=400,
            placeholder="Copy and paste the text you want to summarize here...",
            label_visibility="collapsed"
        )
        
        # Configuration options
        st.subheader("Summary Settings")
        col_a, col_b = st.columns(2)
        with col_a:
            max_length = st.slider("Max Length", 50, 300, 150, help="Maximum length of the summary")
        with col_b:
            min_length = st.slider("Min Length", 10, 150, 50, help="Minimum length of the summary")
    
    with col2:
        st.subheader("AI Summary")
        
        if st.button("Generate Summary", type="primary", use_container_width=True):
            if not input_text.strip():
                st.warning("Please enter some text to summarize.")
            elif len(input_text.split()) < 50:
                st.warning("Please provide longer text for better summarization (at least 50 words).")
            else:
                try:
                    # Show progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Analyzing text structure...")
                    progress_bar.progress(25)
                    
                    status_text.text("Identifying key points...")
                    progress_bar.progress(50)
                    
                    status_text.text("Generating concise summary...")
                    progress_bar.progress(75)
                    
                    # Generate summary
                    summary = summarizer(
                        input_text,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=False,  # Better for consistent results
                        truncation=True
                    )
                    
                    progress_bar.progress(100)
                    status_text.empty()
                    progress_bar.empty()
                    
                    # Display the summary in a nice box
                    st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                    st.write(summary[0]['summary_text'])
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show statistics
                    st.metric("Original Text", f"{len(input_text.split())} words")
                    st.metric("Summary", f"{len(summary[0]['summary_text'].split())} words")
                    st.metric("Reduction", f"{round((1 - len(summary[0]['summary_text'].split()) / len(input_text.split())) * 100)}%")
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.info("Try using slightly shorter text or adjusting the length parameters.")
        else:
            st.info("👈 Enter text and click 'Generate Summary' to see the magic!")
            
            # Placeholder for example
            st.markdown("""
            **Example output will appear here:**
            > The article discusses the latest advancements in AI technology, highlighting 
            > how new models are becoming more efficient and accessible. Researchers are 
            > focusing on making AI systems that can understand context better while 
            > requiring less computational power.
            """)

if __name__ == "__main__":
    main()
