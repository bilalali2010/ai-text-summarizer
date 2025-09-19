# ✨ AI Text Summarizer

A free, professional AI-powered text summarization web app built with Streamlit and Hugging Face's Transformers library. This project was created for **Day 53** of the 100 Days of Code challenge, focusing on NLP and Summarization Models.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app) *<!-- Replace with your actual deployed link -->*

## 🚀 Features

- **Advanced AI Model**: Utilizes Google's `google/pegasus-cnn_dailymail` model, specifically trained for high-quality news and article summarization.
- **Interactive Web Interface**: Clean, user-friendly interface built with Streamlit.
- **Adjustable Summary Length**: Customize the summary with `min_length` and `max_length` sliders.
- **Real-time Statistics**: See the word count and reduction percentage after summarization.
- **100% Free**: Deployed on Streamlit Community Cloud using free-tier resources.

## 📸 Demo

*(You can add a screenshot later after deployment)*
<!-- ![App Screenshot](screenshot.png) -->

## 🛠️ Installation & Local Run

Want to run this locally? Follow these steps:

1.  **Clone the repository**
    ```bash
    git clone https://github.com/bilaiali2010/ai-text-summarizer.git
    cd ai-text-summarizer
    ```

2.  **Create a virtual environment (recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit app**
    ```bash
    streamlit run summarizer_app.py
    ```
    The application will open in your browser at `http://localhost:8501`.

## ☁️ Deployment

This app is designed for easy, free deployment on Streamlit Community Cloud.

1.  Ensure your code is pushed to a **Public** GitHub repository.
2.  Go to [share.streamlit.io](https://share.streamlit.io/).
3.  Sign in with your GitHub account.
4.  Click "New app", select your repository, branch (`main`), and main file path (`summarizer_app.py`).
5.  Click **"Deploy"**! Your app will have a public URL like `https://ai-text-summarizer.streamlit.app`.

## 🧠 How It Works

The core functionality uses the `pipeline` API from the 🤗 Transformers library:

```python
from transformers import pipeline
summarizer = pipeline("summarization", model="google/pegasus-cnn_dailymail")
summary = summarizer(original_text, max_length=150, min_length=50)
