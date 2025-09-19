import streamlit as st
from transformers import pipeline

# Page configuration
st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="✨",
    layout="wide"
)

# Custom CSS to improve the appearance
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .summary-box {
        background-color: #f8f9fa;
        padding: 25px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-top: 20px;
        font-size: 1.1em;
        line-height: 1.6;
    }
    .metric-card {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        color: #6c757d;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown('<p class="main-header">✨ AI Text Summarizer</p>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; margin-bottom: 2rem;'>
    <p>This tool uses a fine-tuned <strong>BART model</strong> for efficient and high-quality text summarization.</p>
    <p>Perfect for summarizing articles, reports, documents, and any long-form text.</p>
</div>
""", unsafe_allow_html=True)

# Cache the model loading to avoid reloading on every interaction
@st.cache_resource(show_spinner="Loading the AI model... Please wait.")
def load_summarizer():
    """Load the summarization pipeline with an efficient model for free tier."""
    try:
        # Using a lighter, more efficient model that works well on free tier
        model_name = "Falconsai/text_summarization"
        
        summarizer = pipeline(
            "summarization",
            model=model_name,
            tokenizer=model_name,
            framework="pt",  # Use PyTorch
        )
        return summarizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def main():
    # Initialize the summarizer
    summarizer = load_summarizer()
    
    if summarizer is None:
        st.error("Failed to load the summarization model. Please try again later.")
        return
    
    st.success("✅ AI model loaded successfully!")
    
    # Create two columns for input and output
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("📝 Input Text")
        input_text = st.text_area(
            "Paste your text here:",
            height=350,
            placeholder="Copy and paste the text you want to summarize here... (Minimum 100 words for best results)",
            label_visibility="collapsed",
            help="The longer and more structured the text, the better the summary will be."
        )
        
        # Configuration options
        st.subheader("⚙️ Summary Settings")
        
        col_a, col_b = st.columns(2)
        with col_a:
            max_length = st.slider(
                "Max Length", 
                50, 300, 150, 
                help="Maximum length of the summary in words"
            )
        with col_b:
            min_length = st.slider(
                "Min Length", 
                30, 150, 50, 
                help="Minimum length of the summary in words"
            )
    
    with col2:
        st.subheader("🤖 AI Summary")
        
        if st.button("🚀 Generate Summary", type="primary", use_container_width=True):
            if not input_text.strip():
                st.warning("⚠️ Please enter some text to summarize.")
            else:
                words = input_text.split()
                if len(words) < 50:
                    st.warning("📋 For better results, please provide at least 50 words.")
                
                try:
                    # Show progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("📖 Reading and analyzing text...")
                    progress_bar.progress(25)
                    
                    status_text.text("🔍 Identifying key concepts...")
                    progress_bar.progress(50)
                    
                    status_text.text("✍️ Generating concise summary...")
                    progress_bar.progress(75)
                    
                    # Generate summary
                    summary_result = summarizer(
                        input_text,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=False,
                        truncation=True
                    )
                    
                    progress_bar.progress(100)
                    status_text.empty()
                    progress_bar.empty()
                    
                    summary_text = summary_result[0]['summary_text']
                    
                    # Display the summary in a nice box
                    st.markdown("### 📄 Your Summary")
                    st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                    st.write(summary_text)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show statistics
                    st.markdown("### 📊 Statistics")
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    
                    with col_stat1:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Original Words", len(words))
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col_stat2:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Summary Words", len(summary_text.split()))
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col_stat3:
                        reduction = round((1 - len(summary_text.split()) / len(words)) * 100)
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Reduction", f"{reduction}%")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ An error occurred during summarization: {str(e)}")
                    st.info("💡 Try using slightly shorter text or adjusting the length parameters.")
        else:
            # Placeholder content
            st.info("👈 Enter text on the left and click 'Generate Summary' to begin!")
            
            st.markdown("""
            <div class="summary-box">
            <h4>Example Output:</h4>
            <p>The article discusses advancements in renewable energy technology, highlighting how solar and wind power are becoming more efficient and cost-effective. Researchers are developing new materials that increase energy conversion rates while reducing manufacturing costs. These innovations are making sustainable energy solutions more accessible to developing countries and contributing to global efforts to combat climate change.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p>Built with ❤️ using <a href="https://streamlit.io/" target="_blank">Streamlit</a> and 
        <a href="https://huggingface.co/" target="_blank">Hugging Face Transformers</a></p>
        <p>Part of the 100 Days of Code Challenge • Day 53: Summarization Models</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
