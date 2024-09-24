import numpy as np
import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from PIL import Image
import plotly.express as px
from textblob import TextBlob
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
import swifter
from wordcloud import  WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

# Make sure to download these if running for the first time
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
st.set_page_config(layout="wide")

st.image("newLogo.png",width=250)
with st.sidebar:
    selected_option = st.selectbox(
        'Choose a plot:',
        ['All','WordCloud','TextLength',"Top 10 Words",'SentimentAnalysis', 'Bigrams','t-SNE']
    )

@st.cache_data
def load_data():
    return pd.read_csv("Suicide_Detection.csv")

data = load_data()
data['cleaned_text'] = data['cleaned_text'].astype(str)  # Convert to string if necessary
data['cleaned_text'] = data['cleaned_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in ['im', 'ive', 'dont','one','cant','like','even','jake']]))

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

    # Choose class to look at
    clas=st.radio(
        "Choose Class",
        ["Both","Suicide","Non-Suicide"]
    )

    data['cleaned_text'] = data['cleaned_text'].astype(str)
    if selected_option == "All" and clas=="Both":
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

        plot1,plot2=st.columns([0.5,0.5])

        with plot1:
            data['text_length'] = data['cleaned_text'].apply(lambda x: len(x.split()))
            # Create a Plotly Express histogram

            suicide_data = data[data['class'] == 'suicide']
            fig_suicide = px.histogram(
            suicide_data, 
            x="text_length", 
            histfunc="count", 
            title="Text Length Distribution for Suicide Class",
            width=420
            )
            st.plotly_chart(fig_suicide)

       
        with plot2:
            non_suicide_data = data[data['class'] == 'non-suicide']
            fig_non_suicide = px.histogram(
            non_suicide_data, 
            x="text_length", 
            histfunc="count", 
            title="Text Length Distribution for Non-Suicide Class",
            width=420
            )
            st.plotly_chart(fig_non_suicide)

        plot12,plot22=st.columns([0.5,0.5])
        def plot_top_words(data, title):
            vectorizer = CountVectorizer(max_features=10)  # Limit to top 10 features for simplicity
            X = vectorizer.fit_transform(data['cleaned_text'])

            word_counts = X.toarray().sum(axis=0)
            words = vectorizer.get_feature_names_out()

            # Create DataFrame
            word_df = pd.DataFrame({'word': words, 'count': word_counts})
            word_df = word_df.sort_values(by='count', ascending=False)  # Sort by descending count

            # Create a Plotly Express bar chart
            fig = px.bar(
                word_df, 
                x="count", 
                y="word", 
                title=title,
                orientation='h',  # Horizontal bar chart for readability
                width=420
            )
            
            return fig

        with plot12:
            fig_suicide = plot_top_words(suicide_data, "Top 10 Most Frequent Words in Suicide Class")
            st.plotly_chart(fig_suicide)
        with plot22:
            fig_non_suicide = plot_top_words(non_suicide_data, "Top 10 Most Frequent Words in Non-Suicide Class")
            st.plotly_chart(fig_non_suicide)

        sent1,sent2=st.columns([0.5,0.5])
        nltk.download('vader_lexicon')

        # Initialize VADER sentiment analyzer
        sid = SentimentIntensityAnalyzer()

        # Function to compute sentiment score
        def classify_sentiment(text):
            sentiment_score = sid.polarity_scores(text)['compound']
            if sentiment_score >= 0.05:
                return 'positive'
            elif sentiment_score <= -0.05:
                return 'negative'
            else:
                return 'neutral'

        # Apply sentiment classification
        data['sentiment'] = data['cleaned_text'].apply(classify_sentiment)
        suicide_data = data[data['class'] == 'suicide']
        non_suicide_data = data[data['class'] == 'non-suicide']

        def plot_sentiment_distribution(data, title):
            sentiment_counts = data['sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['sentiment', 'count']

            # Create Plotly bar chart
            fig = px.bar(
                sentiment_counts, 
                x='sentiment', 
                y='count', 
                title=title, 
                color='sentiment',
                category_orders={'sentiment': ['positive', 'neutral', 'negative']},
                width=420
            )
            
            return fig

        with sent1:
            fig_suicide_sentiment = plot_sentiment_distribution(suicide_data, "Sentiment Distribution for Suicide Class")
            st.plotly_chart(fig_suicide_sentiment)
        with sent2:
            fig_non_suicide_sentiment = plot_sentiment_distribution(non_suicide_data, "Sentiment Distribution for Non-Suicide Class")
            st.plotly_chart(fig_non_suicide_sentiment)

        big1,big2=st.columns([0.5,0.5])
        def create_bigram_plot(data):
            # Create a bigram vectorizer
            bigram_vectorizer = CountVectorizer(ngram_range=(2, 2), max_features=10)
            X_bigrams = bigram_vectorizer.fit_transform(data['cleaned_text'])

            bigram_counts = X_bigrams.toarray().sum(axis=0)
            bigrams = bigram_vectorizer.get_feature_names_out()

            # Create DataFrame for bigrams
            bigram_df = pd.DataFrame({'bigram': bigrams, 'count': bigram_counts})
            bigram_df = bigram_df.sort_values(by='count', ascending=False)

            # Create a Plotly Express bar chart
            fig = px.bar(
                bigram_df,
                x='count',
                y='bigram',
                title=f'Top 10 Bigrams for Class: {data["class"].iloc[0]}',
                width=420
            )
            return fig
        with big1:
            st.plotly_chart(create_bigram_plot(suicide_data))
        with big2:
            st.plotly_chart(create_bigram_plot(non_suicide_data))

        tsne1,tsne2=st.columns([0.5,0.5])
        def plot_tsne(data, class_label):
            # Vectorize text using TF-IDF
            vectorizer = TfidfVectorizer(max_features=100)
            X = vectorizer.fit_transform(data['cleaned_text']).toarray()

            # Perform t-SNE
            tsne = TSNE(n_components=2, random_state=42)
            X_tsne = tsne.fit_transform(X)

            # Add t-SNE components to DataFrame
            data['tsne_1'] = X_tsne[:, 0]
            data['tsne_2'] = X_tsne[:, 1]

            # Create Plotly scatter plot
            fig = px.scatter(
                data, 
                x='tsne_1', 
                y='tsne_2', 
                title=f't-SNE Plot of Text Embeddings for {class_label} Class',
                labels={'tsne_1': 'TSNE Component 1', 'tsne_2': 'TSNE Component 2'},
                width=420,
                # height=500
            )

            return fig
        with tsne1:
            fig_suicide = plot_tsne(suicide_data, 'Suicide')
            st.plotly_chart(fig_suicide)
        with tsne2:
            fig_non_suicide = plot_tsne(non_suicide_data, 'Non-Suicide')
            st.plotly_chart(fig_non_suicide)
            
    elif selected_option == "WordCloud" and clas=="Both":
        # suicide_text = ' '.join(data[data['class'] == 'suicide']['cleaned_text'])
        valid_suicide_text = data[data['class'] == 'suicide'][data['cleaned_text'].str.isnumeric() == False]['cleaned_text']
        suicide_text = ' '.join(valid_suicide_text)
        suicide_wordcloud = WordCloud(width=420).generate(suicide_text)

        non_suicide_text = ' '.join(data[data['class'] == 'non-suicide']['cleaned_text'])
        non_suicide_wordcloud = WordCloud(width=420).generate(non_suicide_text)
        wordBoth1,wordBoth2=st.columns([0.5,0.5])
        with wordBoth1:
            # Displaying the word clouds in Streamlit
            st.write(f"""
            <h6>Word Cloud for Suicide Texts</h6>
            """,unsafe_allow_html=True)
            fig_suicide, ax_suicide = plt.subplots(figsize=(10, 5))
            ax_suicide.imshow(suicide_wordcloud, interpolation='bilinear')
            ax_suicide.axis('off')
            st.pyplot(fig_suicide)  # Display the plot using st.pyplot
        with wordBoth2:
            # Word Cloud for Non-Suicide Texts
            st.write(f"""
            <h6>Word Cloud for Non-Suicide Texts</h6>""",unsafe_allow_html=True)
            fig_non_suicide, ax_non_suicide = plt.subplots(figsize=(10, 5))
            ax_non_suicide.imshow(non_suicide_wordcloud, interpolation='bilinear')
            ax_non_suicide.axis('off')
            st.pyplot(fig_non_suicide)  # Display the plot using st.pyplot
    elif selected_option == "WordCloud" and clas=="Suicide":
        valid_suicide_text = data[data['class'] == 'suicide'][data['cleaned_text'].str.isnumeric() == False]['cleaned_text']
        suicide_text = ' '.join(valid_suicide_text)
        suicide_wordcloud = WordCloud(width=420).generate(suicide_text)  
        st.write(f"""
        <h6>Word Cloud for Suicide Texts</h6>
        """,unsafe_allow_html=True)
        fig_suicide, ax_suicide = plt.subplots(figsize=(10, 5))
        ax_suicide.imshow(suicide_wordcloud, interpolation='bilinear')
        ax_suicide.axis('off')
        st.pyplot(fig_suicide)  # Display the plot using st.pyplot  


    elif selected_option == "WordCloud" and clas=="Non-Suicide":
        non_suicide_text = ' '.join(data[data['class'] == 'non-suicide']['cleaned_text'])
        non_suicide_wordcloud = WordCloud(width=420).generate(non_suicide_text)
        wordBoth1,wordBoth2=st.columns([0.5,0.5])
        st.write(f"""
        <h6>Word Cloud for Non-Suicide Texts</h6>""",unsafe_allow_html=True)
        fig_non_suicide, ax_non_suicide = plt.subplots(figsize=(10, 5))
        ax_non_suicide.imshow(non_suicide_wordcloud, interpolation='bilinear')
        ax_non_suicide.axis('off')
        st.pyplot(fig_non_suicide)  # Display the plot using st.pyplot

    elif selected_option == "Text Length" and clas=="Both":
        length1,length2=st.columns([0.5,0.5])
        with length1:
            data['text_length'] = data['cleaned_text'].apply(lambda x: len(x.split()))
            # Create a Plotly Express histogram

            suicide_data = data[data['class'] == 'suicide']
            fig_suicide = px.histogram(
            suicide_data, 
            x="text_length", 
            histfunc="count", 
            title="Text Length Distribution for Suicide Class",
            width=420
            )
            st.plotly_chart(fig_suicide)

       
        with length2:
            non_suicide_data = data[data['class'] == 'non-suicide']
            fig_non_suicide = px.histogram(
            non_suicide_data, 
            x="text_length", 
            histfunc="count", 
            title="Text Length Distribution for Non-Suicide Class",
            width=420
            )
            st.plotly_chart(fig_non_suicide)
    elif selected_option == "Text Length" and clas=="Suicide":
            data['text_length'] = data['cleaned_text'].apply(lambda x: len(x.split()))
            # Create a Plotly Express histogram
            suicide_data = data[data['class'] == 'suicide']
            fig_suicide = px.histogram(
            suicide_data, 
            x="text_length", 
            histfunc="count", 
            title="Text Length Distribution for Suicide Class",
            width=420
            )
            st.plotly_chart(fig_suicide)
    elif selected_option == "Text Length" and clas=="Non-Suicide":
            non_suicide_data = data[data['class'] == 'non-suicide']
            fig_non_suicide = px.histogram(
            non_suicide_data, 
            x="text_length", 
            histfunc="count", 
            title="Text Length Distribution for Non-Suicide Class",
            width=420
            )
            st.plotly_chart(fig_non_suicide)

    elif selected_option == "Top 10 Words" and clas=="Both":
        top1,top2=st.columns([0.5,0.5])
        def plot_top_words(data, title):
            vectorizer = CountVectorizer(max_features=10)  # Limit to top 10 features for simplicity
            X = vectorizer.fit_transform(data['cleaned_text'])

            word_counts = X.toarray().sum(axis=0)
            words = vectorizer.get_feature_names_out()

            # Create DataFrame
            word_df = pd.DataFrame({'word': words, 'count': word_counts})
            word_df = word_df.sort_values(by='count', ascending=False)  # Sort by descending count

            # Create a Plotly Express bar chart
            fig = px.bar(
                word_df, 
                x="count", 
                y="word", 
                title=title,
                orientation='h',  # Horizontal bar chart for readability
                width=420
            )
            
            return fig
        suicide_data = data[data['class'] == 'suicide']
        non_suicide_data = data[data['class'] == 'non-suicide']
        with top1:
            fig_suicide = plot_top_words(suicide_data, "Top 10 Most Frequent Words in Suicide Class")
            st.plotly_chart(fig_suicide)
        with top2:
            fig_non_suicide = plot_top_words(non_suicide_data, "Top 10 Most Frequent Words in Non-Suicide Class")
            st.plotly_chart(fig_non_suicide)
    
    elif selected_option == "Top 10 Words" and clas=="Suicide":
        def plot_top_words(data, title):
            vectorizer = CountVectorizer(max_features=10)  # Limit to top 10 features for simplicity
            X = vectorizer.fit_transform(data['cleaned_text'])

            word_counts = X.toarray().sum(axis=0)
            words = vectorizer.get_feature_names_out()

            # Create DataFrame
            word_df = pd.DataFrame({'word': words, 'count': word_counts})
            word_df = word_df.sort_values(by='count', ascending=False)  # Sort by descending count

            # Create a Plotly Express bar chart
            fig = px.bar(
                word_df, 
                x="count", 
                y="word", 
                title=title,
                orientation='h',  # Horizontal bar chart for readability
                width=800
            )
            
            return fig
        suicide_data = data[data['class'] == 'suicide']
        fig_suicide = plot_top_words(suicide_data, "Top 10 Most Frequent Words in Suicide Class")
        st.plotly_chart(fig_suicide)

    elif selected_option == "Top 10 Words" and clas=="Non-Suicide":
        def plot_top_words(data, title):
            vectorizer = CountVectorizer(max_features=10)  # Limit to top 10 features for simplicity
            X = vectorizer.fit_transform(data['cleaned_text'])

            word_counts = X.toarray().sum(axis=0)
            words = vectorizer.get_feature_names_out()

            # Create DataFrame
            word_df = pd.DataFrame({'word': words, 'count': word_counts})
            word_df = word_df.sort_values(by='count', ascending=False)  # Sort by descending count

            # Create a Plotly Express bar chart
            fig = px.bar(
                word_df, 
                x="count", 
                y="word", 
                title=title,
                orientation='h',  # Horizontal bar chart for readability
                width=800
            )
            
            return fig
        non_suicide_data = data[data['class'] == 'non-suicide']
        fig_non_suicide = plot_top_words(non_suicide_data, "Top 10 Most Frequent Words in Non-Suicide Class")
        st.plotly_chart(fig_non_suicide)

    elif selected_option=="SentimentAnalysis" and clas=="Both":
        sent12,sent22=st.columns([0.5,0.5])
        nltk.download('vader_lexicon')

        # Initialize VADER sentiment analyzer
        sid = SentimentIntensityAnalyzer()

        # Function to compute sentiment score
        def classify_sentiment(text):
            sentiment_score = sid.polarity_scores(text)['compound']
            if sentiment_score >= 0.05:
                return 'positive'
            elif sentiment_score <= -0.05:
                return 'negative'
            else:
                return 'neutral'

        # Apply sentiment classification
        data['sentiment'] = data['cleaned_text'].apply(classify_sentiment)
        suicide_data = data[data['class'] == 'suicide']
        non_suicide_data = data[data['class'] == 'non-suicide']

        def plot_sentiment_distribution(data, title):
            sentiment_counts = data['sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['sentiment', 'count']

            # Create Plotly bar chart
            fig = px.bar(
                sentiment_counts, 
                x='sentiment', 
                y='count', 
                title=title, 
                color='sentiment',
                category_orders={'sentiment': ['positive', 'neutral', 'negative']},
                width=420
            )
            
            return fig

        with sent12:
            fig_suicide_sentiment = plot_sentiment_distribution(suicide_data, "Sentiment Distribution for Suicide Class")
            st.plotly_chart(fig_suicide_sentiment)
        with sent22:
            fig_non_suicide_sentiment = plot_sentiment_distribution(non_suicide_data, "Sentiment Distribution for Non-Suicide Class")
            st.plotly_chart(fig_non_suicide_sentiment)


    elif selected_option=="SentimentAnalysis" and clas=="Suicide":
        nltk.download('vader_lexicon')

        # Initialize VADER sentiment analyzer
        sid = SentimentIntensityAnalyzer()
        # Function to compute sentiment score
        def classify_sentiment(text):
            sentiment_score = sid.polarity_scores(text)['compound']
            if sentiment_score >= 0.05:
                return 'positive'
            elif sentiment_score <= -0.05:
                return 'negative'
            else:
                return 'neutral'

        # Apply sentiment classification
        data['sentiment'] = data['cleaned_text'].apply(classify_sentiment)
        suicide_data = data[data['class'] == 'suicide']
        def plot_sentiment_distribution(data, title):
            sentiment_counts = data['sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['sentiment', 'count']

            # Create Plotly bar chart
            fig = px.bar(
                sentiment_counts, 
                x='sentiment', 
                y='count', 
                title=title, 
                color='sentiment',
                category_orders={'sentiment': ['positive', 'neutral', 'negative']},
                width=420
            )
            
            return fig
        fig_suicide_sentiment = plot_sentiment_distribution(suicide_data, "Sentiment Distribution for Suicide Class")
        st.plotly_chart(fig_suicide_sentiment)


    elif selected_option=="SentimentAnalysis" and clas=="Non-Suicide":
        nltk.download('vader_lexicon')

        # Initialize VADER sentiment analyzer
        sid = SentimentIntensityAnalyzer()
        # Function to compute sentiment score
        def classify_sentiment(text):
            sentiment_score = sid.polarity_scores(text)['compound']
            if sentiment_score >= 0.05:
                return 'positive'
            elif sentiment_score <= -0.05:
                return 'negative'
            else:
                return 'neutral'

        # Apply sentiment classification
        data['sentiment'] = data['cleaned_text'].apply(classify_sentiment)
        non_suicide_data = data[data['class'] == 'non-suicide']
        def plot_sentiment_distribution(data, title):
            sentiment_counts = data['sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['sentiment', 'count']

            # Create Plotly bar chart
            fig = px.bar(
                sentiment_counts, 
                x='sentiment', 
                y='count', 
                title=title, 
                color='sentiment',
                category_orders={'sentiment': ['positive', 'neutral', 'negative']},
                width=420
            )
            
            return fig
        fig_suicide_sentiment = plot_sentiment_distribution(non_suicide_data, "Sentiment Distribution for Non-Suicide Class")
        st.plotly_chart(fig_suicide_sentiment)

    elif selected_option == "Bigrams" and clas=="Both":
        big12,big22=st.columns([0.5,0.5])
        suicide_data = data[data['class'] == 'suicide']
        non_suicide_data = data[data['class'] == 'non-suicide']
        def create_bigram_plot(data):
            # Create a bigram vectorizer
            bigram_vectorizer = CountVectorizer(ngram_range=(2, 2), max_features=10)
            X_bigrams = bigram_vectorizer.fit_transform(data['cleaned_text'])

            bigram_counts = X_bigrams.toarray().sum(axis=0)
            bigrams = bigram_vectorizer.get_feature_names_out()

            # Create DataFrame for bigrams
            bigram_df = pd.DataFrame({'bigram': bigrams, 'count': bigram_counts})
            bigram_df = bigram_df.sort_values(by='count', ascending=False)

            # Create a Plotly Express bar chart
            fig = px.bar(
                bigram_df,
                x='count',
                y='bigram',
                title=f'Top 10 Bigrams for Class: {data["class"].iloc[0]}',
                width=420
            )
            return fig
        with big12:
            st.plotly_chart(create_bigram_plot(suicide_data))
        with big22:
            st.plotly_chart(create_bigram_plot(non_suicide_data))


    elif selected_option == "Bigrams" and clas=="Suicide":
        suicide_data = data[data['class'] == 'suicide']
        def create_bigram_plot(data):
            # Create a bigram vectorizer
            bigram_vectorizer = CountVectorizer(ngram_range=(2, 2), max_features=10)
            X_bigrams = bigram_vectorizer.fit_transform(data['cleaned_text'])

            bigram_counts = X_bigrams.toarray().sum(axis=0)
            bigrams = bigram_vectorizer.get_feature_names_out()

            # Create DataFrame for bigrams
            bigram_df = pd.DataFrame({'bigram': bigrams, 'count': bigram_counts})
            bigram_df = bigram_df.sort_values(by='count', ascending=False)

            # Create a Plotly Express bar chart
            fig = px.bar(
                bigram_df,
                x='count',
                y='bigram',
                title=f'Top 10 Bigrams for Class: {data["class"].iloc[0]}',
                width=800
            )
            return fig
        st.plotly_chart(create_bigram_plot(suicide_data))

    elif selected_option == "Bigrams" and clas=="Non-Suicide":
        non_suicide_data = data[data['class'] == 'non-suicide']
        def create_bigram_plot(data):
            # Create a bigram vectorizer
            bigram_vectorizer = CountVectorizer(ngram_range=(2, 2), max_features=10)
            X_bigrams = bigram_vectorizer.fit_transform(data['cleaned_text'])

            bigram_counts = X_bigrams.toarray().sum(axis=0)
            bigrams = bigram_vectorizer.get_feature_names_out()

            # Create DataFrame for bigrams
            bigram_df = pd.DataFrame({'bigram': bigrams, 'count': bigram_counts})
            bigram_df = bigram_df.sort_values(by='count', ascending=False)

            # Create a Plotly Express bar chart
            fig = px.bar(
                bigram_df,
                x='count',
                y='bigram',
                title=f'Top 10 Bigrams for Class: {data["class"].iloc[0]}',
                width=800
            )
            return fig
        st.plotly_chart(create_bigram_plot(non_suicide_data))

    elif selected_option=="t-SNE" and clas=="Both":
        suicide_data = data[data['class'] == 'suicide']
        non_suicide_data = data[data['class'] == 'non-suicide']
        tsne12,tsne22=st.columns([0.5,0.5])
        def plot_tsne(data, class_label):
            # Vectorize text using TF-IDF
            vectorizer = TfidfVectorizer(max_features=100)
            X = vectorizer.fit_transform(data['cleaned_text']).toarray()

            # Perform t-SNE
            tsne = TSNE(n_components=2, random_state=42)
            X_tsne = tsne.fit_transform(X)

            # Add t-SNE components to DataFrame
            data['tsne_1'] = X_tsne[:, 0]
            data['tsne_2'] = X_tsne[:, 1]

            # Create Plotly scatter plot
            fig = px.scatter(
                data, 
                x='tsne_1', 
                y='tsne_2', 
                title=f't-SNE Plot of Text Embeddings for {class_label} Class',
                labels={'tsne_1': 'TSNE Component 1', 'tsne_2': 'TSNE Component 2'},
                width=420,
                # height=500
            )

            return fig
        with tsne12:
            fig_suicide = plot_tsne(suicide_data, 'Suicide')
            st.plotly_chart(fig_suicide)
        with tsne22:
            fig_non_suicide = plot_tsne(non_suicide_data, 'Non-Suicide')
            st.plotly_chart(fig_non_suicide)


    elif selected_option=="t-SNE" and clas=="Suicide":
        suicide_data = data[data['class'] == 'suicide']
        def plot_tsne(data, class_label):
            # Vectorize text using TF-IDF
            vectorizer = TfidfVectorizer(max_features=100)
            X = vectorizer.fit_transform(data['cleaned_text']).toarray()

            # Perform t-SNE
            tsne = TSNE(n_components=2, random_state=42)
            X_tsne = tsne.fit_transform(X)

            # Add t-SNE components to DataFrame
            data['tsne_1'] = X_tsne[:, 0]
            data['tsne_2'] = X_tsne[:, 1]

            # Create Plotly scatter plot
            fig = px.scatter(
                data, 
                x='tsne_1', 
                y='tsne_2', 
                title=f't-SNE Plot of Text Embeddings for {class_label} Class',
                labels={'tsne_1': 'TSNE Component 1', 'tsne_2': 'TSNE Component 2'},
                width=800,
                # height=500
            )

            return fig
        fig_suicide = plot_tsne(suicide_data, 'Suicide')
        st.plotly_chart(fig_suicide)

    elif selected_option=="t-SNE" and clas=="Non-Suicide":
        non_suicide_data = data[data['class'] == 'non-suicide']
        def plot_tsne(data, class_label):
            # Vectorize text using TF-IDF
            vectorizer = TfidfVectorizer(max_features=100)
            X = vectorizer.fit_transform(data['cleaned_text']).toarray()

            # Perform t-SNE
            tsne = TSNE(n_components=2, random_state=42)
            X_tsne = tsne.fit_transform(X)

            # Add t-SNE components to DataFrame
            data['tsne_1'] = X_tsne[:, 0]
            data['tsne_2'] = X_tsne[:, 1]

            # Create Plotly scatter plot
            fig = px.scatter(
                data, 
                x='tsne_1', 
                y='tsne_2', 
                title=f't-SNE Plot of Text Embeddings for {class_label} Class',
                labels={'tsne_1': 'TSNE Component 1', 'tsne_2': 'TSNE Component 2'},
                width=800,
                # height=500
            )

            return fig
        fig_non_suicide = plot_tsne(non_suicide_data, 'Non-Suicide')
        st.plotly_chart(fig_non_suicide)








