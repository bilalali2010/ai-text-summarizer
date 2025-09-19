import streamlit as st
from transformers import pipeline
import time

# ========== PAGE CONFIGURATION ==========
st.set_page_config(
    page_title="SummifyAI | Advanced Text Summarization",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CUSTOM CSS FOR PROFESSIONAL UI ==========
st.markdown("""
<style>
    /* Main background styling */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Professional header */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(45deg, #ffffff, #e0e0e0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        padding-top: 1rem;
    }
    
    /* Subheader styling */
    .sub-header {
        color: #ffffff;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        opacity: 0.9;
    }
    
    /* Content container */
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        margin-bottom: 2rem;
    }
    
    /* Metric cards styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Summary box styling */
    .summary-box {
        background: #f8f9ff;
        border-left: 4px solid #667eea;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
    }
    
    /* Slider styling */
    .stSlider {
        margin: 1rem 0;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        color: rgba(255, 255, 255, 0.7);
        padding: 1.5rem;
        font-size: 0.9rem;
        margin-top: 2rem;
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# ========== HEADER SECTION ==========
st.markdown('<p class="main-header">SummifyAI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced AI-Powered Text Summarization</p>', unsafe_allow_html=True)

# ========== MAIN CONTENT CONTAINER ==========
with st.container():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # ========== MODEL LOADING ==========
    @st.cache_resource(show_spinner=False)
    def load_summarizer():
        """Load the summarization pipeline."""
        try:
            summarizer = pipeline(
                "summarization",
                model="Falconsai/text_summarization",
                framework="pt",
            )
            return summarizer
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None

    # ========== TWO COLUMN LAYOUT ==========
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 📝 Input Text")
        input_text = st.text_area(
            "",
            height=300,
            placeholder="Paste your article, report, or any text you want to summarize here...",
            label_visibility="collapsed",
            help="For best results, use well-structured text of 100+ words"
        )
        
        st.markdown("---")
        st.markdown("### ⚙️ Customization")
        
        col_a, col_b = st.columns(2)
        with col_a:
            max_length = st.slider(
                "**Max Length**", 
                50, 300, 150,
                help="Maximum length of summary in words"
            )
        with col_b:
            min_length = st.slider(
                "**Min Length**", 
                30, 150, 50,
                help="Minimum length of summary in words"
            )

    with col2:
        st.markdown("### 🤖 AI Summary")
        
        if st.button("🚀 Generate Summary", type="primary", use_container_width=True):
            if not input_text.strip():
                st.warning("Please enter some text to summarize.")
            else:
                words = input_text.split()
                if len(words) < 50:
                    st.warning("For optimal results, provide at least 50 words.")
                
                summarizer = load_summarizer()
                
                if summarizer:
                    try:
                        # Progress animation
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i in range(100):
                            progress_bar.progress(i + 1)
                            if i < 25:
                                status_text.text("📖 Reading and analyzing text...")
                            elif i < 50:
                                status_text.text("🔍 Identifying key concepts...")
                            elif i < 75:
                                status_text.text("✨ Crafting optimal summary...")
                            else:
                                status_text.text("✅ Finalizing results...")
                            time.sleep(0.01)
                        
                        # Generate summary
                        summary_result = summarizer(
                            input_text,
                            max_length=max_length,
                            min_length=min_length,
                            do_sample=False,
                            truncation=True
                        )
                        
                        progress_bar.empty()
                        status_text.empty()
                        
                        summary_text = summary_result[0]['summary_text']
                        
                        # Display summary
                        st.markdown("### 📄 Generated Summary")
                        st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                        st.write(summary_text)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Statistics
                        st.markdown("### 📊 Analysis")
                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                        
                        with col_stat1:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric("Original Words", f"{len(words):,}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col_stat2:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric("Summary Words", f"{len(summary_text.split()):,}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col_stat3:
                            reduction = round((1 - len(summary_text.split()) / len(words)) * 100)
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric("Reduction", f"{reduction}%")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                else:
                    st.error("Model failed to load. Please try again.")
        else:
            # Placeholder content
            st.info("💡 **Ready to summarize!** Paste your text on the left and click the button to generate a professional summary.")
            
            st.markdown("---")
            st.markdown("### 🎯 Example Output")
            st.markdown("""
            <div class="summary-box">
            <p><strong>Renewable energy adoption is accelerating globally as solar and wind power become cost-competitive with traditional fossil fuels. Technological advancements have dramatically improved efficiency while reducing production costs. Many countries are investing heavily in green infrastructure to meet climate goals and ensure energy security.</strong></p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close main container

# ========== FOOTER ==========
st.markdown("""
<div class="footer">
    <p>Built with ❤️ using Streamlit & Hugging Face Transformers</p>
    <p>© 2024 SummifyAI • Part of 100 Days of Code Challenge</p>
</div>
""", unsafe_allow_html=True)

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h2>⚡ SummifyAI</h2>
        <p>Professional Text Summarization</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📋 How to Use")
    st.markdown("""
    1. **Paste** your text in the input area
    2. **Adjust** length settings if needed
    3. **Click** Generate Summary
    4. **Review** your AI-powered summary
    """)
    
    st.markdown("---")
    st.markdown("### 🎯 Best For")
    st.markdown("""
    • Research papers
    • News articles
    • Business reports
    • Legal documents
    • Blog content
    • Technical manuals
    """)
    
    st.markdown("---")
    st.markdown("### 🔧 Technology")
    st.markdown("""
    - **AI Model:** BART Transformer
    - **Framework:** Hugging Face
    - **Platform:** Streamlit
    - **Deployment:** Streamlit Cloud
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center;'>
        <p>Need help? Contact support</p>
    </div>
    """, unsafe_allow_html=True)
