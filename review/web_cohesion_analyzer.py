import streamlit as st
import requests
from bs4 import BeautifulSoup
import spacy
import warnings
import matplotlib.pyplot as plt

# Function to get web content
def get_web_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    content = ' '.join([p.get_text() for p in paragraphs])
    return content

# Function to perform NLP-based cohesion analysis
def cohesion_analysis(text):
    # Load spaCy English model
    nlp = spacy.load("en_core_web_sm")

    # Process the text using spaCy
    doc = nlp(text)

    # Convert generator to list
    sentences = list(doc.sents)

    # Print the number of sentences for debugging
    num_sentences = len(sentences)
    st.write(f'**Number of sentences**: {num_sentences}')

    # Suppress the warning message
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Calculate semantic similarity between consecutive sentences
        cohesion_scores = []
        for sent1, sent2 in zip(sentences, sentences[1:]):
            similarity = sent1.similarity(sent2)
            cohesion_scores.append(similarity)

    # Calculate average cohesion score
    if cohesion_scores:
        average_cohesion = sum(cohesion_scores) / len(cohesion_scores)
        st.write(f'**Average Semantic Cohesion**: {average_cohesion:.2f}')
        return cohesion_scores, sentences
    else:
        st.write('Not enough sentences for cohesion analysis.')
        return [], []

# Plot cohesion scores
def plot_cohesion_scores(scores, sentences):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(scores) + 1), scores, marker='o', linestyle='-', color='b')
    ax.set_xlabel('Sentence Pair Index')
    ax.set_ylabel('Cohesion Score')
    ax.set_title('Cohesion Scores between Consecutive Sentences')
    ax.set_xticks(range(1, len(scores) + 1))
    ax.set_xticklabels([f'{i}-{i+1}' for i in range(1, len(sentences))], rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

# Streamlit UI
st.title('Web Page Cohesion Analyzer')
st.write("Analyze the cohesion of consecutive sentences in a web article.")

web_link = st.text_input('Enter the web link', '')

if web_link:
    try:
        st.write('Fetching web content...')
        web_content = get_web_content(web_link)
        st.write('Performing cohesion analysis...')
        cohesion_scores, sentences = cohesion_analysis(web_content)
        if cohesion_scores:
            plot_cohesion_scores(cohesion_scores, sentences)
    except Exception as e:
        st.error(f'Error: {e}')
else:
    st.write('Please enter a valid web link.')
