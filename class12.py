""" import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import string

# Sample text
text = "Text mining is the process of deriving meaningful information from natural language text."

# Tokenize the text
tokens = word_tokenize(text)

# Remove punctuation
tokens = [token.lower() for token in tokens if token not in string.punctuation]

# Remove stopwords
stop_words = set(stopwords.words('english'))
tokens = [token for token in tokens if token not in stop_words]

# Calculate frequency distribution of words
freq_dist = FreqDist(tokens)

# Print the most common words and their frequencies
print("Most common words:")
for word, frequency in freq_dist.most_common(5):
    print(f"{word}: {frequency}")
 """
# Required Libraries
import pandas as pd
import requests
from bs4 import BeautifulSoup
import spacy

# Load SpaCy Model for NLP
nlp = spacy.load("en_core_web_sm")

# Function for Data Extraction
def extract_data(url=None, file_path=None):
    if url:
        # Web scraping
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        text_data = soup.get_text(separator="\n")
        return text_data
    elif file_path:
        # Reading a local file
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    else:
        raise ValueError("Provide a URL or file path for data extraction!")

# Function for NLP Processing
def process_text(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    word_freq = {}
    for token in doc:
        if token.is_alpha and not token.is_stop:
            word_freq[token.text.lower()] = word_freq.get(token.text.lower(), 0) + 1
    return entities, word_freq

# Main Script
if __name__ == "__main__":
    # Example: Extract text from a URL or a local file
    # url = "https://en.wikipedia.org/wiki/Natural_language_processing"
    # text = extract_data(url=url)
    file_path = "example.txt"  # Replace with a valid file path
    text = extract_data(file_path=file_path)

    # Process the extracted text
    named_entities, word_frequency = process_text(text)

    # Display Results
    print("Named Entities:")
    for entity, label in named_entities:
        print(f"{entity} ({label})")

    print("\nWord Frequency:")
    word_freq_df = pd.DataFrame(word_frequency.items(), columns=["Word", "Frequency"]).sort_values(by="Frequency", ascending=False)
    print(word_freq_df.head(10))
