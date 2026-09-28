import random
import re
from collections import defaultdict, Counter


class BigramModel:
    """Simple bigram language model: learns which word tends to follow each word."""

    def __init__(self, corpus):
        self.bigram_counts = defaultdict(Counter)
        self.vocab = set()
        self._train(corpus)

    @staticmethod
    def _tokenize(text):
        # Lowercase and keep only words (drops punctuation)
        return re.findall(r"\b\w+\b", text.lower())

    def _train(self, corpus):
        for text in corpus:
            tokens = self._tokenize(text)
            self.vocab.update(tokens)
            for w1, w2 in zip(tokens[:-1], tokens[1:]):
                self.bigram_counts[w1][w2] += 1

    def next_word_probabilities(self, word):
        """P(next_word | word) = count(word, next_word) / count(word, *)"""
        counts = self.bigram_counts.get(word.lower())
        if not counts:
            return {}
        total = sum(counts.values())
        return {w: c / total for w, c in counts.items()}

    def generate_text(self, start_word, length):
        """Generate `length` words starting from `start_word`, sampling by bigram probability."""
        current = start_word.lower()
        words = [current]

        for _ in range(length - 1):
            probs = self.next_word_probabilities(current)
            if not probs:
                break  # no known next word, stop early
            next_words = list(probs.keys())
            weights = list(probs.values())
            current = random.choices(next_words, weights=weights, k=1)[0]
            words.append(current)

        return " ".join(words)