import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from heapq import nlargest


def preprocess_text(text):
    sentences = sent_tokenize(text)
    words = [word.lower() for sentence in sentences for word in word_tokenize(sentence)]
    stopwords_set = set(stopwords.words('english'))
    words = [word for word in words if word not in stopwords_set]
    return words


def calculate_word_frequencies(words):
    frequency = defaultdict(int)
    for word in words:
        frequency[word] += 1
    max_frequency = max(frequency.values())
    for word in frequency.keys():
        frequency[word] = (frequency[word] / max_frequency)
    return frequency


def score_sentences(sentences, frequency):
    sentence_scores = defaultdict(int)
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in frequency:
                sentence_scores[sentence] += frequency[word]
    return sentence_scores


def generate_summary(text, num_sentences):
    words = preprocess_text(text)
    frequency = calculate_word_frequencies(words)
    sentences = sent_tokenize(text)
    sentence_scores = score_sentences(sentences, frequency)
    summary_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    return ' '.join(summary_sentences)


# Example usage
text = input("Enter the paragraph to be summarized: ")
summary = generate_summary(text, 2)
print("Summary:")
print(summary)
