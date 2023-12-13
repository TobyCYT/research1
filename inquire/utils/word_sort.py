# Use nltk to sort the words in to verb, noun, adjective, and adverb

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def word_sort(caption):
    # Lowercase the caption
    caption = caption.lower()

    # Tokenize the caption
    tokens = word_tokenize(caption)

    # Remove the stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [w for w in tokens if w.isalpha()]

    # Sort the words
    verb = []
    noun = []
    adjective = []
    adverb = []
    for word in tokens:
        if wordnet.synsets(word, pos=wordnet.VERB):
            verb.append(word)
        elif wordnet.synsets(word, pos=wordnet.NOUN):
            noun.append(word)
        elif wordnet.synsets(word, pos=wordnet.ADJ):
            adjective.append(word)
        elif wordnet.synsets(word, pos=wordnet.ADV):
            adverb.append(word)
        else:
            noun.append(word)

    # convert list into tuples containing the word and its frequency
    verb = nltk.FreqDist(verb)
    noun = nltk.FreqDist(noun)
    adjective = nltk.FreqDist(adjective)
    adverb = nltk.FreqDist(adverb)

    # Convert the frequency count into a percentage of that given part of speech
    total = len(tokens)
    for word in verb:
        verb[word] = verb[word] / total
    for word in noun:
        noun[word] = noun[word] / total
    for word in adjective:
        adjective[word] = adjective[word] / total
    for word in adverb:
        adverb[word] = adverb[word] / total



    return verb, noun, adjective, adverb

def main():
    caption = "A man playing a guitar on stage with a microphone with a crowd of people watching. The man is passionately immersed in his performance."
    print(word_sort(caption))

if __name__ == '__main__':
    main()
