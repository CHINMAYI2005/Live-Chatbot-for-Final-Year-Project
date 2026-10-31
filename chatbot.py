"""
Live Chatbot using Natural Language Processing (NLP) & TF-IDF
Final Year Project Implementation
"""

import io
import os
import random
import string
import warnings
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Suppress unnecessary warnings
warnings.filterwarnings('ignore')

# Ensure necessary NLTK corpora are downloaded
for resource in ['punkt', 'punkt_tab', 'wordnet', 'omw-1.4']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else f'corpora/{resource}')
    except (LookupError, IndexError):
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

class NLPChatbot:
    """
    Retrieval-Based NLP Chatbot using Lemmatization, TF-IDF Vectorization,
    and Cosine Similarity.
    """
    def __init__(self, corpus_path="corpus.txt"):
        self.corpus_path = corpus_path
        self.lemmer = WordNetLemmatizer()
        self.remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
        self.sent_tokens = []
        self.word_tokens = []
        
        # Predefined greeting patterns
        self.GREETING_INPUTS = (
            "hello", "hi", "greetings", "sup", "what's up", "hey", 
            "hey there", "good morning", "good evening", "howdy"
        )
        self.GREETING_RESPONSES = [
            "Hi there! How can I help you today?",
            "Hello! What would you like to learn about?",
            "Hey! Feel free to ask me any question.",
            "Greetings! How may I assist you with NLP or AI topics?",
            "Hello! I am ready to answer your questions."
        ]

        # Predefined gratitude patterns
        self.THANKS_INPUTS = ("thanks", "thank you", "thx", "much appreciated", "thank you so much")
        self.THANKS_RESPONSES = [
            "You're welcome! Glad I could help.",
            "Anytime! Let me know if you need anything else.",
            "My pleasure!",
            "Happy to help!"
        ]

        # Load and prepare knowledge corpus
        self.load_corpus()

    def load_corpus(self):
        """Loads and tokenizes the corpus text file."""
        if not os.path.exists(self.corpus_path):
            raise FileNotFoundError(f"Corpus file not found at: {self.corpus_path}")

        with open(self.corpus_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw_data = f.read().lower()

        # Tokenize into sentences and words
        self.sent_tokens = nltk.sent_tokenize(raw_data)
        self.word_tokens = nltk.word_tokenize(raw_data)

    def lem_tokens(self, tokens):
        """Returns lemmatized tokens."""
        return [self.lemmer.lemmatize(token) for token in tokens]

    def lem_normalize(self, text):
        """Removes punctuation, lowercases, and lemmatizes text."""
        return self.lem_tokens(nltk.word_tokenize(text.lower().translate(self.remove_punct_dict)))

    def check_greeting(self, sentence):
        """Checks if user input is a greeting and returns a canned response."""
        for word in sentence.split():
            if word.lower() in self.GREETING_INPUTS:
                return random.choice(self.GREETING_RESPONSES)
        # Check phrase match
        cleaned = sentence.lower().strip()
        if cleaned in self.GREETING_INPUTS:
            return random.choice(self.GREETING_RESPONSES)
        return None

    def check_thanks(self, sentence):
        """Checks if user input is an expression of gratitude."""
        cleaned = sentence.lower().strip()
        for phrase in self.THANKS_INPUTS:
            if phrase in cleaned:
                return random.choice(self.THANKS_RESPONSES)
        return None

    def generate_response(self, user_response):
        """
        Calculates TF-IDF matrix and cosine similarity to find the most relevant
        sentence from the corpus.
        """
        user_response = user_response.lower().strip()

        # Check for gratitude
        thanks_reply = self.check_thanks(user_response)
        if thanks_reply:
            return thanks_reply

        # Check for greeting
        greeting_reply = self.check_greeting(user_response)
        if greeting_reply:
            return greeting_reply

        # Append user query to sentence list for vector comparison
        self.sent_tokens.append(user_response)

        try:
            # Generate TF-IDF vectors
            tfidf_vectorizer = TfidfVectorizer(tokenizer=self.lem_normalize, stop_words='english')
            tfidf_matrix = tfidf_vectorizer.fit_transform(self.sent_tokens)

            # Compute cosine similarity between user query (last vector) and corpus sentences
            vals = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
            idx = vals.argsort()[0][-1]
            flat = vals.flatten()
            flat.sort()
            req_tfidf = flat[-2]

            # Pop the appended user query
            self.sent_tokens.pop()

            # Threshold check for relevance
            if req_tfidf == 0 or req_tfidf < 0.15:
                return "I am sorry, but I do not have sufficient information in my knowledge base to answer that. Could you please rephrase?"
            else:
                return self.sent_tokens[idx].capitalize()

        except Exception as e:
            if len(self.sent_tokens) > 0 and self.sent_tokens[-1] == user_response:
                self.sent_tokens.pop()
            return f"An error occurred while generating a response: {str(e)}"

def run_cli():
    """Runs the chatbot in command-line interface mode."""
    print("=" * 65)
    print("   🤖 LIVE NLP CHATBOT - FINAL YEAR PROJECT")
    print("=" * 65)
    print("Type your questions below. Type 'bye', 'quit', or 'exit' to end.")
    print("-" * 65)

    chatbot = NLPChatbot(corpus_path="corpus.txt")
    print("Chatbot initialized successfully! Knowledge corpus loaded.\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["bye", "exit", "quit", "goodbye"]:
                print("Bot: Goodbye! Have a wonderful day ahead!")
                break

            response = chatbot.generate_response(user_input)
            print(f"Bot: {response}\n")

        except (KeyboardInterrupt, EOFError):
            print("\nBot: Session closed. Goodbye!")
            break

if __name__ == "__main__":
    run_cli()
