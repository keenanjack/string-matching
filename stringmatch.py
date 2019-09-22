import sys
import re
from sklearn.feature_extraction.text import TfidfVectorizer

strings_to_match = 
lookup_strings = 

def ngrams(string, n=3):
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

print(ngrams('McDonalds'))