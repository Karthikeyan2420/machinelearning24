import nltk
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
