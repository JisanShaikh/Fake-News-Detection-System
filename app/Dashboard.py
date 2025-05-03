import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud
import time

# Set page config
st.set_page_config(
    page_title="Fake News Detection",
    layout="wide",
    page_icon="📰",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling with vibrant colors
st.markdown("""
    <style>
        :root {
            --primary: #4361ee;
            --secondary: #3f37c9;
            --accent1: #4895ef;
            --accent2: #4cc9f0;
            --light: #f8f9fa;
            --dark: #212529;
            --success: #4bb543;
            --danger: #ff3333;
            --warning: #ffc107;
        }
        
        .main {background-color: #f0f2f6;}
        .reportview-container .main .block-container {padding-top: 2rem;}
        .sidebar .sidebar-content {
            background-color: #4361ee;
            background-image: linear-gradient(to bottom, #4361ee, #3f37c9);
        }
        .sidebar .sidebar-content .sidebar-section {
            color: white;
        }
        h1 {color: #3f37c9;}
        h2 {color: #4361ee;}
        h3 {color: #4895ef;}
        .st-bb {background-color: transparent;}
        .st-at {background-color: transparent;}
        .footer {font-size: 0.8rem; color: #666; text-align: center; margin-top: 2rem;}
        .positive {color: #4bb543; font-weight: bold;}
        .negative {color: #ff3333; font-weight: bold;}
        .feature-box {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            border-left: 5px solid #4895ef;
        }
        .stButton>button {
            background-color: #4361ee;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            border: none;
        }
        .stButton>button:hover {
            background-color: #3f37c9;
            color: white;
        }
        .stTextInput>div>div>input {
            border-radius: 5px;
            padding: 0.5rem;
            background-color: white;
        }
        .stTextArea>div>div>textarea {
            border-radius: 5px;
            padding: 0.5rem;
            background-color: white;
        }
        .stRadio>div {
            flex-direction: row;
            align-items: center;
        }
        .stRadio>div>label {
            margin-right: 15px;
        }
        .stProgress>div>div>div>div {
            background-color: #4361ee;
        }
        /* Remove white backgrounds from text elements */
        .stMarkdown, .stAlert, .stText {
            background-color: transparent !important;
        }
        /* Fix for dataframe background */
        .stDataFrame {
            background-color: white;
        }
        /* Make sure text is visible in all elements */
        .st-bw, .st-bx, .st-by {
            background-color: transparent !important;
            color: inherit !important;
        }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        data_true = pd.read_csv("True.csv")
        data_fake = pd.read_csv("Fake.csv")
        
        # Add class labels
        data_true["class"] = 1  # 1 for true news
        data_fake["class"] = 0  # 0 for fake news
        
        # Merge datasets
        data_merge = pd.concat([data_fake, data_true], axis=0)
        
        # Keep only text and class columns
        data = data_merge[['text', 'class']].sample(frac=1, random_state=42).reset_index(drop=True)
        
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

data = load_data()

# Data preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('\s+', ' ', text).strip()
    return text

# Train model
@st.cache_resource
def train_model(data):
    try:
        # Preprocess text
        data['text'] = data['text'].apply(preprocess_text)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            data['text'], data['class'], 
            test_size=0.2, random_state=42, stratify=data['class']
        )
        
        # Vectorize text
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
        tfidf_train = tfidf_vectorizer.fit_transform(X_train) 
        tfidf_test = tfidf_vectorizer.transform(X_test)
        
        # Train classifier
        pac = PassiveAggressiveClassifier(max_iter=50)
        pac.fit(tfidf_train, y_train)
        
        # Calculate accuracy
        y_pred = pac.predict(tfidf_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return pac, tfidf_vectorizer, accuracy, X_test, y_test, y_pred
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None, None, None, None, None

# Sidebar for navigation
st.sidebar.title("📰 Fake News Detector")
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2965/2965879.png", width=100)
st.sidebar.markdown("""
    <div style="margin-top: 20px; margin-bottom: 30px;">
        <p style="color: #3f37c9;">This app uses machine learning to classify news articles as real or fake based on their text content.</p>
    </div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("Navigation", ["🏠 Home", "📊 Data Exploration", "🤖 Model Testing", "📈 Performance Metrics"], key="nav")

# Main content
if page == "🏠 Home":
    st.title("Fake News Detection Dashboard")
    st.markdown("""
        <div class="feature-box">
            <h3 style="color: #3f37c9;">Welcome to the Fake News Detection System</h3>
            <p style="color: #000000;">In today's digital age, the spread of misinformation has become a significant challenge. 
            This application helps identify potentially fake news articles using machine learning techniques.</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="feature-box">
                <h4 style="color: #4895ef;">📋 Dataset Information</h4>
                <p style="color: #000000;">The dataset contains news articles labeled as either 'True' or 'Fake'. 
                We use text analysis to classify news articles based on their content.</p>
                <ul style="color: #000000;">
                    <li>Total articles: {:,}</li>
                    <li>True news: {:,}</li>
                    <li>Fake news: {:,}</li>
                </ul>
            </div>
        """.format(
            len(data),
            len(data[data['class'] == 1]),
            len(data[data['class'] == 0])
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="feature-box">
                <h4 style="color: #4895ef;">🔍 How It Works</h4>
                <ol style="color: #000000;">
                    <li>Text preprocessing (cleaning, normalization)</li>
                    <li>Feature extraction (TF-IDF vectorization)</li>
                    <li>Classification (Passive Aggressive Algorithm)</li>
                    <li>Result interpretation</li>
                </ol>
                <p style="color: #000000;">The model achieves approximately 92-95% accuracy on test data.</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="feature-box">
            <h4 style="color: #4895ef;">🚀 Getting Started</h4>
            <p style="color: #000000;">Use the navigation menu on the left to:</p>
            <ul style="color: #000000;">
                <li>Explore the dataset and visualizations</li>
                <li>Test the model with your own news text</li>
                <li>View model performance metrics</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

elif page == "📊 Data Exploration":
    st.title("Data Exploration")
    
    if data is not None:
        tab1, tab2 = st.tabs(["📋 Dataset Overview", "📈 Visualizations"])
        
        with tab1:
            st.subheader("Dataset Overview")
            st.dataframe(data.head(10), height=400)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Basic Statistics**")
                st.write(f"Total samples: {len(data):,}")
                st.write(f"True news: {len(data[data['class'] == 1]):,} ({len(data[data['class'] == 1])/len(data)*100:.1f}%)")
                st.write(f"Fake news: {len(data[data['class'] == 0]):,} ({len(data[data['class'] == 0])/len(data)*100:.1f}%)")
            
            with col2:
                st.markdown("**Text Length Statistics**")
                data['text_length'] = data['text'].apply(len)
                st.write(f"Average length: {data['text_length'].mean():.0f} characters")
                st.write(f"Minimum length: {data['text_length'].min():.0f} characters")
                st.write(f"Maximum length: {data['text_length'].max():.0f} characters")
        
        with tab2:
            st.subheader("Data Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Class Distribution**")
                fig, ax = plt.subplots(figsize=(6, 4))
                data['class'].value_counts().plot(kind='pie', autopct='%1.1f%%', 
                                                labels=['Fake News', 'True News'], 
                                                colors=['#ff3333', '#4bb543'], ax=ax)
                ax.set_ylabel('')
                st.pyplot(fig)
            
            with col2:
                st.markdown("**Text Length Distribution**")
                fig, ax = plt.subplots(figsize=(6, 4))
                # Convert class to string for proper plotting
                plot_data = data.copy()
                plot_data['class'] = plot_data['class'].map({0: 'Fake News', 1: 'True News'})
                sns.boxplot(x='class', y='text_length', data=plot_data, 
                          palette={'Fake News': '#ff3333', 'True News': '#4bb543'}, ax=ax)
                ax.set_xlabel('')
                ax.set_ylabel('Text Length (characters)')
                st.pyplot(fig)
            
            st.markdown("**Detailed Text Length Analysis**")
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            
            # Fake news length
            fake_lengths = data[data['class'] == 0]['text_length']
            sns.histplot(fake_lengths, bins=50, color='#ff3333', ax=ax[0], kde=True)
            ax[0].set_title('Fake News Text Length', color='#000000')
            ax[0].set_xlabel('Length (characters)')
            ax[0].set_ylabel('Count')
            
            # True news length
            true_lengths = data[data['class'] == 1]['text_length']
            sns.histplot(true_lengths, bins=50, color='#4bb543', ax=ax[1], kde=True)
            ax[1].set_title('True News Text Length', color='000000')
            ax[1].set_xlabel('Length (characters)')
            
            st.pyplot(fig)
    else:
        st.error("Data not loaded. Please check your data files.")

elif page == "🤖 Model Testing":
    st.title("Fake News Detection Model")
    
    if data is not None:
        with st.spinner('Training the model... This may take a few moments.'):
            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.02)
                progress_bar.progress(percent_complete + 1)
            
            model, vectorizer, accuracy, X_test, y_test, y_pred = train_model(data)
        
        st.success(f"Model trained successfully! (Test Accuracy: {accuracy*100:.1f}%)")
        
        st.markdown("""
            <div class="feature-box">
                <h4 style="color: #4895ef;">Test the Model with Your Own Text</h4>
                <p style="color: #000000;">Enter a news article in the text box below to check if it's likely to be real or fake.</p>
            </div>
        """, unsafe_allow_html=True)
        
        user_input = st.text_area("News Article Text:", 
                                "Paste news article here...", 
                                height=300)
        
        analyze_button = st.button("Analyze Article", type="primary")
        
        if analyze_button:
            if user_input.strip() == "Paste news article here..." or not user_input.strip():
                st.warning("Please enter a news article to analyze.")
            else:
                with st.spinner('Analyzing text...'):
                    # Preprocess and predict
                    processed_text = preprocess_text(user_input)
                    vectorized_text = vectorizer.transform([processed_text])
                    prediction = model.predict(vectorized_text)
                    proba = model._predict_proba_lr(vectorized_text)[0]
                    
                    # Display result with animation
                    result_placeholder = st.empty()
                    
                    if prediction[0] == 0:
                        result_placeholder.markdown("""
                            <div style="background-color: #ffebee; padding: 20px; border-radius: 10px; border-left: 5px solid #ff3333;">
                                <h3 class="negative" style="color: #000000;">⚠️ This news article is likely FAKE.</h3>
                                <p style="color: #000000;">Confidence: <span class="negative">{:.1f}%</span></p>
                            </div>
                        """.format(max(proba)*100), unsafe_allow_html=True)
                    else:
                        result_placeholder.markdown("""
                            <div style="background-color: #e8f5e9; padding: 20px; border-radius: 10px; border-left: 5px solid #4bb543;">
                                <h3 class="positive" style="color: #000000;">✅ This news article is likely TRUE.</h3>
                                <p style="color: #000000;">Confidence: <span class="positive">{:.1f}%</span></p>
                            </div>
                        """.format(max(proba)*100), unsafe_allow_html=True)
                    
                    # Show text analysis
                    st.subheader("Text Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Processed Text (First 500 chars)**")
                        st.text(processed_text[:500] + ("..." if len(processed_text) > 500 else ""))
                    
                    with col2:
                        st.markdown("**Key Features**")
                        feature_names = vectorizer.get_feature_names_out()
                        coefs = model.coef_.toarray()[0]
                        top_features = sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)[:10]
                        
                        for feature, weight in top_features:
                            color = "#ff3333" if weight < 0 else "#4bb543"
                            st.markdown(f"<span style='color:{color}'>◼</span> {feature}: {weight:.3f}", unsafe_allow_html=True)
    else:
        st.error("Data not loaded. Please check your data files.")

elif page == "📈 Performance Metrics":
    st.title("Model Performance Metrics")
    
    if data is not None:
        model, vectorizer, accuracy, X_test, y_test, y_pred = train_model(data)
        
        st.markdown(f"""
            <div class="feature-box">
                <h3 style="color: #3f37c9;">Model Evaluation</h3>
                <p style="color: #000000;">The Passive Aggressive Classifier achieved an accuracy of <strong style="color: #4361ee;">{accuracy*100:.1f}%</strong> on the test set.</p>
            </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["📊 Classification Report", "📉 Confusion Matrix"])
        
        with tab1:
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, target_names=['Fake News', 'True News'], output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.highlight_max(axis=0, color='#e6f3ff'))
            
            st.markdown("""
                <div style="margin-top: 20px;">
                    <h4 style="color: #4895ef;">Key Metrics:</h4>
                    <ul style="color: #000000;">
                        <li><strong>Precision</strong>: Percentage of correctly identified instances among all predicted instances for each class</li>
                        <li><strong>Recall</strong>: Percentage of correctly identified instances among all actual instances for each class</li>
                        <li><strong>F1-score</strong>: Harmonic mean of precision and recall</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        with tab2:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            
            fig, ax = plt.subplots(figsize=(6, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Fake News', 'True News'], 
                        yticklabels=['Fake News', 'True News'], ax=ax)
            ax.set_xlabel('Predicted', color='#3f37c9')
            ax.set_ylabel('Actual', color='#3f37c9')
            ax.set_title('Confusion Matrix', color='#3f37c9')
            st.pyplot(fig)
            
            st.markdown("""
                <div style="margin-top: 20px;">
                    <h4 style="color: #4895ef;">Interpretation:</h4>
                    <ul style="color: #000000;">
                        <li><strong>True Positives (TP)</strong>: Correctly identified fake news</li>
                        <li><strong>True Negatives (TN)</strong>: Correctly identified true news</li>
                        <li><strong>False Positives (FP)</strong>: True news incorrectly labeled as fake</li>
                        <li><strong>False Negatives (FN)</strong>: Fake news incorrectly labeled as true</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.error("Data not loaded. Please check your data files.")

# Footer
st.markdown("""
    <div class="footer">
        <hr>
        <p style="color: #000000;">Fake News Detection System • Developed with Streamlit and Scikit-learn</p>
        <p style="color: #000000;">Note: This is a demonstration system. Always verify information from multiple reliable sources.</p>
    </div>
""", unsafe_allow_html=True)