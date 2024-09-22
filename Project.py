import numpy as np
import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from PIL import Image
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import swifter
from wordcloud import  WordCloud
import matplotlib.pyplot as plt

# Make sure to download these if running for the first time
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
st.set_page_config(layout="wide")

st.image("newLogo.png",width=350)


@st.cache_data
def load_data():
    return pd.read_csv("Suicide_Detection.csv")

data = load_data()


# data=data.iloc[:,1:]
st.write("""
#### Suicide and Its Detection in Data Science

**Suicide:**
Suicide is a critical public health issue, involving the intentional end of one's own life. It is often linked to mental health conditions and life stressors.

**Suicide Detection in Data Science:**
Data science helps identify suicide risk by analyzing various data sources:

- **Text Data:** Analyzing social media or online texts using sentiment analysis and NLP to detect signs of distress.
- **Medical Records:** Examining electronic health records for patterns related to mental health and previous attempts.
- **Machine Learning Models:** Using algorithms to classify risk levels and detect unusual behavior patterns.


Data science enhances suicide prevention by identifying risk factors and supporting timely interventions.
""")
tab1, tab2= st.tabs(["Data Visualisation", "Data Modeling"])
with tab1:
    total_rows = len(data)
    total_suicidal_texts = len(data[data['class'] == 'suicide'])
    total_non_suicidal_texts = len(data[data['class'] == 'non-suicide'])

    # Display KPIs using custom HTML
    st.write(f"""
        <div style="display: flex; justify-content: space-between;">
            <div style="padding: 8px; border: 2px solid #ddd; border-radius: 10px; width: 30%;height:6%;border-left:7px solid purple;">
                <p style="font-weight: bold;line-height:1;">Total Rows</p>
                <h4 style="line-height:0">{total_rows}</h4>
                <p style="color: green;">+ 0</p>
            </div>
            <div style="padding: 8px; border: 2px solid #ddd; border-radius: 10px; width: 30%;height:6%;border-left:7px solid purple;">
                <p style="font-weight: bold;line-height:1;">Total Suicidal Text</p>
                <h4 style="line-height:0">{total_suicidal_texts}</h4>
                <p style="color: green;">+ 0</p>
            </div>
            <div style="padding: 8px; border: 2px solid #ddd; border-radius: 10px; width: 30%;height:6%;border-left:7px solid purple;">
                <p style="font-weight: bold;line-height:1;">Total NoSuicide Text</p>
                <h4 style="line-height:0">{total_non_suicidal_texts}</h4>
                <p style="color: green;">+ 0</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.write("<br><br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([0.3,0.3,1])



    # Button in the first column
    with col1:
        if st.button("Suicide"):
            st.write("Suicide button clicked")

    # Button in the second column
    with col2:
        if st.button("Non-Suicide"):
            st.write("Non-Suicide button clicked")

    # Button in the third column
    with col3:
        if st.button("Both"):
            st.write("Both button clicked")

    selected_option = st.selectbox(
        'Choose a plot:',
        ['WordCloud', 'Sentiment Analysis', 'Correlation Plot','All']
    )
    data['cleaned_text'] = data['cleaned_text'].astype(str)
    if selected_option == "All":
        # suicide_text = ' '.join(data[data['class'] == 'suicide']['cleaned_text'])
        valid_suicide_text = data[data['class'] == 'suicide'][data['cleaned_text'].str.isnumeric() == False]['cleaned_text']
        suicide_text = ' '.join(valid_suicide_text)
        suicide_wordcloud = WordCloud(width=800, height=400).generate(suicide_text)

        non_suicide_text = ' '.join(data[data['class'] == 'non-suicide']['cleaned_text'])
        non_suicide_wordcloud = WordCloud(width=800, height=400).generate(non_suicide_text)
        # Plot the word clouds in columns
        word1,word2=st.columns([0.5,0.5])
        with word1:
            # Displaying the word clouds in Streamlit
            st.write(f"""
            <h6>Word Cloud for Suicide Texts</h6>
            """,unsafe_allow_html=True)
            fig_suicide, ax_suicide = plt.subplots(figsize=(10, 5))
            ax_suicide.imshow(suicide_wordcloud, interpolation='bilinear')
            ax_suicide.axis('off')
            st.pyplot(fig_suicide)  # Display the plot using st.pyplot
        with word2:
            # Word Cloud for Non-Suicide Texts
            st.write(f"""
            <h6>Word Cloud for Non-Suicide Texts</h6>""",unsafe_allow_html=True)
            fig_non_suicide, ax_non_suicide = plt.subplots(figsize=(10, 5))
            ax_non_suicide.imshow(non_suicide_wordcloud, interpolation='bilinear')
            ax_non_suicide.axis('off')
            st.pyplot(fig_non_suicide)  # Display the plot using st.pyplot




