import pandas as pd
import string
import nltk
from nltk import word_tokenize, pos_tag
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Read the CSV file into a DataFrame
df = pd.read_csv('dataset/filtered_30words_6sec_train.csv')

# Create an empty dictionary to store word counts
word_count = {}

# Define a set of characters to remove
remove_chars = set(string.punctuation)

# Function to filter for nouns and verbs
def is_noun_verb(pos):
    return pos[0][1] == 'NN' or pos[0][1] == 'VB'

# Iterate through the DataFrame rows
for x in df.index:
    caption = df['name'][x]

    # Tokenize the caption into words (split by spaces)
    words = caption.split()

    # Remove symbols from words and update the word counts
    for word in words:
        # Remove symbols from the word
        word = ''.join(char for char in word if char not in remove_chars)

        # Perform part-of-speech tagging
        pos = pos_tag(word_tokenize(word))
        #print(pos)
        if len(pos) == 0:
            continue

        # Check if the word is a noun or verb and update the word counts
        if is_noun_verb(pos):
            word_count[word] = word_count.get(word, 0) + 1

# Sort the word counts in descending order
sorted_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

# Print the top 100 noun and verb words
count = 0
for word, count in sorted_word_count[:100]:
    print(f'{word}: {count}')
