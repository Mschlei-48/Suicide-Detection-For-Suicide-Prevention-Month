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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import lime
from lime import lime_tabular
import joblib
from lime.lime_text import LimeTextExplainer
from streamlit_cropper import st_cropper
import pytesseract

# Import all the libraries you may need
import pytesseract

# Optionally set the path to tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'



st.set_page_config(layout="wide")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.write(f"<h1>Suicide Detection</h1>",unsafe_allow_html=True)
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
data['cleaned_text'] = data['cleaned_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in ['im', 'ive', 'dont','one','cant','like','even','jake','paulfuck']]))


tab1, tab2= st.tabs(["Data Visualisation", "Data Modeling"])
with tab1:
    total_rows = len(data)
    total_suicidal_texts = len(data[data['class'] == 'suicide'])
    total_non_suicidal_texts = len(data[data['class'] == 'non-suicide'])

    # Display KPIs using custom HTML
    st.write(f"""
        <div style="display: flex; justify-content: space-between;">
            <div style="padding: 8px; border: 2px solid #ddd; border-radius: 10px; width: 30%;height:6%;border-left:7px solid purple;background-color:white;color:black">
                <p style="font-weight: bold;line-height:1;">Total Rows</p>
                <h4 style="line-height:0;color:black">{total_rows}</h4>
                <p style="color: green;">+ 0</p>
            </div>
            <div style="padding: 8px; border: 2px solid #ddd; border-radius: 10px; width: 30%;height:6%;border-left:7px solid purple;background-color:white;color:black">
                <p style="font-weight: bold;line-height:1;">Total Suicidal Text</p>
                <h4 style="line-height:0;color:black;">{total_suicidal_texts}</h4>
                <p style="color: green;">+ 0</p>
            </div>
            <div style="padding: 8px; border: 2px solid #ddd; border-radius: 10px; width: 30%;height:6%;border-left:7px solid purple;background-color:white;color:black;">
                <p style="font-weight: bold;line-height:1;">Total NoSuicide Text</p>
                <h4 style="line-height:0;color:black">{total_non_suicidal_texts}</h4>
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
            title="Text Length Distribution for Suicide Texts",
            nbins=30,
            width=420,
            )
            st.plotly_chart(fig_suicide)

       
        with plot2:
            non_suicide_data = data[data['class'] == 'non-suicide']
            fig_non_suicide = px.histogram(
            non_suicide_data, 
            x="text_length", 
            histfunc="count", 
            title="Text Length Distribution for Non-Suicide Texts",
            width=420,
            nbins=30
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
                width=420,

            )
            fig.update_layout(
            # xaxis_showgrid=True,
             yaxis_showgrid=True
            )
            
            return fig

        with plot12:
            fig_suicide = plot_top_words(suicide_data, "Top 10 Most Frequent Words in Suicide Texts")
            st.plotly_chart(fig_suicide)
        with plot22:
            fig_non_suicide = plot_top_words(non_suicide_data, "Top 10 Most Frequent Words in Non-Suicide Texts")
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
            fig.update_layout(
            # xaxis_showgrid=True,
             yaxis_showgrid=True
            )
            
            return fig

        with sent1:
            fig_suicide_sentiment = plot_sentiment_distribution(suicide_data, "Sentiment Distribution for Suicide Texts")
            st.plotly_chart(fig_suicide_sentiment)
        with sent2:
            fig_non_suicide_sentiment = plot_sentiment_distribution(non_suicide_data, "Sentiment Distribution for Non-Suicide Texts")
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
                title=f'Top 10 Bigrams for {data["class"].iloc[0]} Texts',
                width=420
            )
            fig.update_layout(
            # xaxis_showgrid=True,
             yaxis_showgrid=True
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
                title=f't-SNE Plot of Text Embeddings for {class_label} Texts',
                labels={'tsne_1': 'TSNE Component 1', 'tsne_2': 'TSNE Component 2'},
                width=420,
                # height=500
            )
            fig.update_layout(
            # xaxis_showgrid=True,
             yaxis_showgrid=True
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
            title="Text Length Distribution for Suicide Texts",
            nbins=30,
            width=420
            )
            st.plotly_chart(fig_suicide)

       
        with length2:
            non_suicide_data = data[data['class'] == 'non-suicide']
            fig_non_suicide = px.histogram(
            non_suicide_data, 
            x="text_length", 
            histfunc="count", 
            title="Text Length Distribution for Non-Suicide Texts",
            nbins=30,
            width=420
            )
            st.plotly_chart(fig_non_suicide)

    # All for Non-Suicide

    elif selected_option=="All" and clas=="Non-Suicide":
        word1,word2=st.columns([0.5,0.5])
        len1,len2=st.columns([0.5,0.5])
        fre1,fre2=st.columns([0.9,0.1])
        with word1:
            non_suicide_data = data[data['class'] == 'non-suicide']
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
                    width=420,

                )
                fig.update_layout(
                # xaxis_showgrid=True,
                yaxis_showgrid=True
                )
                
                return fig
            fig_non_suicide = plot_top_words(non_suicide_data, "Top 10 Most Frequent Words in Non-Suicide Texts")
            st.plotly_chart(fig_non_suicide)
        with word2:
            data['text_length'] = data['cleaned_text'].apply(lambda x: len(x.split()))
            non_suicide_data = data[data['class'] == 'non-suicide']
            fig_non_suicide = px.histogram(
            non_suicide_data, 
            x="text_length", 
            histfunc="count", 
            title="Text Length Distribution for Non-Suicide  Texts",
            nbins=30,
            width=420
            )
            # st.write(f"""
            #  <h6>Text Length Distribution for Non-Suicide Class</h6>""",unsafe_allow_html=True)
            st.plotly_chart(fig_non_suicide)
        with len1:
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
                fig.update_layout(
                # xaxis_showgrid=True,
                yaxis_showgrid=True
                )
                
                return fig
            fig_suicide_sentiment = plot_sentiment_distribution(non_suicide_data, "Sentiment Distribution for Non-Suicide Texts")
            st.plotly_chart(fig_suicide_sentiment)

        with len2:
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
                    title=f'Top 10 Bigrams for Non-Suicide Texts',
                    width=800
                )
                fig.update_layout(
                # xaxis_showgrid=True,
                yaxis_showgrid=True
                )
                return fig
            st.plotly_chart(create_bigram_plot(non_suicide_data))
        with fre1:
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
                    title=f't-SNE Plot of Text Embeddings for {class_label} Texts',
                    labels={'tsne_1': 'TSNE Component 1', 'tsne_2': 'TSNE Component 2'},
                    width=800,
                    # height=500
                )

                return fig
            fig_non_suicide = plot_tsne(non_suicide_data, 'Non-Suicide')
            st.plotly_chart(fig_non_suicide)


# End of all for non-suicide


# All for suicide
    elif selected_option=="All" and clas=="Suicide":
        word1,word2=st.columns([0.5,0.5])
        len1,len2=st.columns([0.5,0.5])
        fre1,fre2=st.columns([0.9,0.1])
        with word1:
            suicide_data = data[data['class'] == 'suicide']
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
                    width=420,

                )
                fig.update_layout(
                # xaxis_showgrid=True,
                yaxis_showgrid=True
                )
                
                return fig
            fig_suicide = plot_top_words(suicide_data, "Top 10 Most Frequent Words in Suicide Texts")
            st.plotly_chart(fig_suicide)
        with word2:
            data['text_length'] = data['cleaned_text'].apply(lambda x: len(x.split()))
            suicide_data = data[data['class'] == 'suicide']
            fig_suicide = px.histogram(
            suicide_data, 
            x="text_length", 
            histfunc="count", 
            title="Text Length Distribution for Suicide Texts",
            nbins=30,
            width=420
            )
            # st.write(f"""
            #  <h6>Text Length Distribution for Non-Suicide Class</h6>""",unsafe_allow_html=True)
            st.plotly_chart(fig_suicide)
        with len1:
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
                fig.update_layout(
                # xaxis_showgrid=True,
                yaxis_showgrid=True
                )
                
                return fig
            fig_suicide_sentiment = plot_sentiment_distribution(suicide_data, "Sentiment Distribution for Suicide Texts")
            st.plotly_chart(fig_suicide_sentiment)

        with len2:
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
                    title=f'Top 10 Bigrams for Suicide Texts',
                    width=420
                )
                fig.update_layout(
                # xaxis_showgrid=True,
                yaxis_showgrid=True
                )
                return fig
            st.plotly_chart(create_bigram_plot(suicide_data))
        with fre1:
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
                    title=f't-SNE Plot of Text Embeddings for {class_label} Texts',
                    labels={'tsne_1': 'TSNE Component 1', 'tsne_2': 'TSNE Component 2'},
                    width=800,
                    # height=500
                )

                return fig
            fig_suicide = plot_tsne(suicide_data, 'Suicide')
            st.plotly_chart(fig_suicide)
    elif selected_option == "TextLength" and clas=="Suicide":
        data['text_length'] = data['cleaned_text'].apply(lambda x: len(x.split()))
        # Create a Plotly Express histogram
        suicide_data = data[data['class'] == 'suicide']
        fig_suicide = px.histogram(
        suicide_data, 
        x="text_length", 
        histfunc="count", 
        title="Text Length Distribution for Suicide Texts",
        nbins=30,
        width=800
        )
        st.plotly_chart(fig_suicide)




    elif selected_option == "TextLength" and clas=="Non-Suicide":
        data['text_length'] = data['cleaned_text'].apply(lambda x: len(x.split()))
        non_suicide_data = data[data['class'] == 'non-suicide']
        fig_non_suicide = px.histogram(
        non_suicide_data, 
        x="text_length", 
        histfunc="count", 
        title="Text Length Distribution for Non-Suicide Texts",
        nbins=30,
        width=800
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
            fig.update_layout(
            # xaxis_showgrid=True,
             yaxis_showgrid=True
            )
            
            return fig
        suicide_data = data[data['class'] == 'suicide']
        non_suicide_data = data[data['class'] == 'non-suicide']
        with top1:
            fig_suicide = plot_top_words(suicide_data, "Top 10 Most Frequent Words in Suicide Texts")
            st.plotly_chart(fig_suicide)
        with top2:
            fig_non_suicide = plot_top_words(non_suicide_data, "Top 10 Most Frequent Words in Non-Suicide Texts")
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
            fig.update_layout(
            # xaxis_showgrid=True,
             yaxis_showgrid=True
            )
            
            return fig
        suicide_data = data[data['class'] == 'suicide']
        fig_suicide = plot_top_words(suicide_data, "Top 10 Most Frequent Words in Suicide Texts")
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
            fig.update_layout(
            # xaxis_showgrid=True,
             yaxis_showgrid=True
            )
            
            return fig
        non_suicide_data = data[data['class'] == 'non-suicide']
        fig_non_suicide = plot_top_words(non_suicide_data, "Top 10 Most Frequent Words in Non-Suicide Texts")
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
            fig.update_layout(
            # xaxis_showgrid=True,
             yaxis_showgrid=True
            )
            
            return fig

        with sent12:
            fig_suicide_sentiment = plot_sentiment_distribution(suicide_data, "Sentiment Distribution for Suicide Texts")
            st.plotly_chart(fig_suicide_sentiment)
        with sent22:
            fig_non_suicide_sentiment = plot_sentiment_distribution(non_suicide_data, "Sentiment Distribution for Non-Suicide Texts")
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
            fig.update_layout(
            # xaxis_showgrid=True,
             yaxis_showgrid=True
            )
            
            return fig
        fig_suicide_sentiment = plot_sentiment_distribution(suicide_data, "Sentiment Distribution for Suicide Texts")
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
            fig.update_layout(
            # xaxis_showgrid=True,
             yaxis_showgrid=True
            )
            
            return fig
        fig_suicide_sentiment = plot_sentiment_distribution(non_suicide_data, "Sentiment Distribution for Non-Suicide Texts")
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
                title=f'Top 10 Bigrams for {data["class"].iloc[0]} Texts',
                width=420
            )
            fig.update_layout(
            # xaxis_showgrid=True,
             yaxis_showgrid=True
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
                title=f'Top 10 Bigrams for {data["class"].iloc[0]} Texts',
                width=800
            )
            fig.update_layout(
            # xaxis_showgrid=True,
             yaxis_showgrid=True
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
                title=f'Top 10 Bigrams for {data["class"].iloc[0]} Texts',
                width=800
            )
            fig.update_layout(
            # xaxis_showgrid=True,
             yaxis_showgrid=True
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
                title=f't-SNE Plot of Text Embeddings for {class_label} Texts',
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
                title=f't-SNE Plot of Text Embeddings for {class_label} Texts',
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
                title=f't-SNE Plot of Text Embeddings for {class_label} Texts',
                labels={'tsne_1': 'TSNE Component 1', 'tsne_2': 'TSNE Component 2'},
                width=800,
                # height=500
            )

            return fig
        fig_non_suicide = plot_tsne(non_suicide_data, 'Non-Suicide')
        st.plotly_chart(fig_non_suicide)
with tab2:

    # st.write(f"""
    # <h3>Text Classification</h3>
    # """,alow_unsafe_html=True)
    st.write(f"<h3>Explainable Text Classification With LIME and Random Forest</h3>",unsafe_allow_html=True)
    # st.write("Upload a text file/image/copy text and paste in the text box below to get a classification.")
    file = st.file_uploader("Upload a text file/image to get a classification.", type=["txt", "png", "jpeg", "jpg"])

    def check_file_type(uploaded_file):
        if uploaded_file is not None:
            # Get the file extension
            file_extension = uploaded_file.name.split('.')[-1].lower()
            return file_extension
        return None

    file_type = check_file_type(file)
    if file is not None:
        if file_type in ['jpg', 'jpeg', 'png']:
            image = Image.open(file)
            
            # Show the cropping tool only
            cropped_image = st_cropper(image, aspect_ratio=None)

            # Display the cropped image
            st.image(cropped_image, caption="Cropped Image", use_column_width=True)
            if st.button("Classify Extracted Text"):
                text_data = pytesseract.image_to_string(cropped_image)
                def clean_text(text):
                    # Convert text to lowercase
                    text = text.lower()

                    # Remove URLs
                    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

                    # Remove HTML tags
                    text = re.sub(r'<.*?>', '', text)

                    # Remove punctuation and special characters
                    text = re.sub(r'[^a-zA-Z\s]', '', text)

                    # Remove numbers
                    text = re.sub(r'\d+', '', text)

                    # Tokenize text
                    words = text.split()

                    # Remove stopwords
                    stop_words = set(stopwords.words('english'))
                    words = [word for word in words if word not in stop_words]

                    # Lemmatize words
                    lemmatizer = WordNetLemmatizer()
                    words = [lemmatizer.lemmatize(word) for word in words]

                    # Rejoin words into a single string
                    cleaned_text = ' '.join(words)

                    return cleaned_text
                df = pd.DataFrame([text_data], columns=['Text'])
                df["Text"] = df["Text"].apply(clean_text)
                cleaned_text_data=df.loc[0,"Text"]
                st.write("Extracted Text:")
                st.text(cleaned_text_data)
                # Get modeling results
                # Load the saved model and vectorizer
                model = joblib.load('text_model.pkl')
                vectorizer = joblib.load('vectorizer.pkl')

                # Preprocess the input text for prediction
                def preprocess_input(text):
                    # Vectorize the text using the loaded vectorizer
                    X_text = vectorizer.transform([text])
                    return X_text

                # Process the input data
                input_data = preprocess_input(cleaned_text_data)

                # Get the prediction
                prediction = model.predict(input_data)
                st.write(f'Prediction: {prediction[0]}')

                # Initialize LimeTextExplainer (text-based explanation)
                explainer = LimeTextExplainer(class_names=['non-suicide', 'suicide'])  # Adjust class names as necessary

                # Define a prediction function for LIME that works with raw text input
                def predict_proba(texts):
                    text_vectors = vectorizer.transform(texts)
                    return model.predict_proba(text_vectors)

                # Explain the instance using LimeTextExplainer
                exp = explainer.explain_instance(text_data, predict_proba)

                # Display LIME explanations in a readable format
                exp_html = exp.as_html()  # Convert explanation to HTML
                st.components.v1.html(exp_html, height=800)
        else:
            bytes_data = file.read()
            text_data = bytes_data.decode("utf-8")  # Decode bytes to string
            df = pd.DataFrame([text_data], columns=['Text'])

            def clean_text(text):
                # Convert text to lowercase
                text = text.lower()

                # Remove URLs
                text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

                # Remove HTML tags
                text = re.sub(r'<.*?>', '', text)

                # Remove punctuation and special characters
                text = re.sub(r'[^a-zA-Z\s]', '', text)

                # Remove numbers
                text = re.sub(r'\d+', '', text)

                # Tokenize text
                words = text.split()

                # Remove stopwords
                stop_words = set(stopwords.words('english'))
                words = [word for word in words if word not in stop_words]

                # Lemmatize words
                lemmatizer = WordNetLemmatizer()
                words = [lemmatizer.lemmatize(word) for word in words]

                # Rejoin words into a single string
                cleaned_text = ' '.join(words)

                return cleaned_text
            df["Text"] = df["Text"].apply(clean_text)
            cleaned_text_data=df.loc[0,"Text"]
            # Display the text content in the app
            st.text_area("File content",  cleaned_text_data, height=100)
            model = joblib.load('text_model.pkl')
            vectorizer = joblib.load('vectorizer.pkl')

            # Preprocess the input text for prediction
            def preprocess_input(text):
                # Vectorize the text using the loaded vectorizer
                X_text = vectorizer.transform([text])
                return X_text

            # Process the input data
            input_data = preprocess_input(cleaned_text_data)

            # Get the prediction
            prediction = model.predict(input_data)
            st.write(f'Prediction: {prediction[0]}')

            # Initialize LimeTextExplainer (text-based explanation)
            explainer = LimeTextExplainer(class_names=['non-suicide', 'suicide'])  # Adjust class names as necessary

            # Define a prediction function for LIME that works with raw text input
            def predict_proba(texts):
                text_vectors = vectorizer.transform(texts)
                return model.predict_proba(text_vectors)

            # Explain the instance using LimeTextExplainer
            exp = explainer.explain_instance(text_data, predict_proba)

            # Display LIME explanations in a readable format
            exp_html = exp.as_html()  # Convert explanation to HTML
            st.components.v1.html(exp_html, height=800)




    # if file is not None:
    #     # Read the file content
    #     bytes_data = file.read()
    #     text_data = bytes_data.decode("utf-8")  # Decode bytes to string
        
    #     # Display the text content in the app
    #     st.text_area("File content", text_data, height=100)
        
    #     if st.button("Get Classification"):
    #         # Load the saved model and vectorizer
    #         model = joblib.load('text_model.pkl')
    #         vectorizer = joblib.load('vectorizer.pkl')

    #         # Preprocess the input text for prediction
    #         def preprocess_input(text):
    #             # Vectorize the text using the loaded vectorizer
    #             X_text = vectorizer.transform([text])
    #             return X_text

    #         # Process the input data
    #         input_data = preprocess_input(text_data)

    #         # Get the prediction
    #         prediction = model.predict(input_data)
    #         st.write(f'Prediction: {prediction[0]}')

    #         # Initialize LimeTextExplainer (text-based explanation)
    #         explainer = LimeTextExplainer(class_names=['non-suicide', 'suicide'])  # Adjust class names as necessary

    #         # Define a prediction function for LIME that works with raw text input
    #         def predict_proba(texts):
    #             text_vectors = vectorizer.transform(texts)
    #             return model.predict_proba(text_vectors)

    #         # Explain the instance using LimeTextExplainer
    #         exp = explainer.explain_instance(text_data, predict_proba)

    #         # Display LIME explanations in a readable format
    #         exp_html = exp.as_html()  # Convert explanation to HTML
    #         st.components.v1.html(exp_html, height=800)
















  