import os # debug

import sys # for command-line args

# Imports for preprocessing
import re
import tokenize
import io

# Imports for cosine similarity
from collections import Counter
import math



# HELPER METHODS

# 0. PREPROCESSING METHODS (for text and code)
def preprocess_text(text: str) -> str:
    # Convert text to lowercase, remove punctuation, and tokenize
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = text.split()
    return tokens

def preprocess_code(code: str) -> str:
    # Remove comments and whitespace, process tokens for code
    code = re.sub(r'//.*?$|/\*.*?\*/|#.*', '', code, flags=re.S | re.M)  # Removing comments
    code_tokens = re.split(r'\s+', code.strip())  # Split by whitespace
    return code_tokens

# 1. N-GRAM TOKENIZATION
def n_grams(tokens, n=3):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

# 2. COSINE SIMILARITY
def cosine_similarity(doc1, doc2):
    vec1 = Counter(doc1)
    vec2 = Counter(doc2)
    
    # Dot product
    intersection = set(vec1.keys()) & set(vec2.keys())
    dot_product = sum([vec1[x] * vec2[x] for x in intersection])
    
    # Magnitude of vectors
    magnitude1 = math.sqrt(sum([val**2 for val in vec1.values()]))
    magnitude2 = math.sqrt(sum([val**2 for val in vec2.values()]))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)


# LEVENSHTEIN DISTANCE
def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

# GET SCORES
def plagiarism_score(doc1, doc2, is_code=False):
    if is_code:
        tokens1 = preprocess_code(doc1)
        tokens2 = preprocess_code(doc2)
    else:
        tokens1 = preprocess_text(doc1)
        tokens2 = preprocess_text(doc2)
    
    # N-gram similarity
    ngrams1 = n_grams(tokens1)
    ngrams2 = n_grams(tokens2)
    
    cosine_sim = cosine_similarity(ngrams1, ngrams2)
    levenshtein_dist = levenshtein_distance(' '.join(tokens1), ' '.join(tokens2))
    
    return cosine_sim, levenshtein_dist

# EVALUATE SCORES
def is_plagiarism(cosine_sim, levenshtein_dist, doc_length, cosine_threshold=0.7, levenshtein_threshold_ratio=0.1):    
    # Check if cosine similarity is above the threshold
    cosine_check = cosine_sim >= cosine_threshold
    
    # Check if Levenshtein distance is below a threshold based on document length
    levenshtein_threshold = levenshtein_threshold_ratio * doc_length
    levenshtein_check = levenshtein_dist <= levenshtein_threshold
    
    # If both checks pass, we flag it as potential plagiarism
    return cosine_check and levenshtein_check

# CENTRAL METHOD
def checkPlagiarism(originalFile, testFile):
    cosine_sim, levenshtein_dist = plagiarism_score(originalFile, testFile)
    doc_length = max(len(originalFile), len(testFile))

    # Determine if plagiarism has occurred
    plagiarized = is_plagiarism(cosine_sim, levenshtein_dist, doc_length)

    if plagiarized:
        return 1
    return 0
  

if __name__ == "__main__":
  try:
    originalFile = sys.argv[1]
    testFile = sys.argv[2]
  except IndexError: 
    print("Enter file path")
    exit()  

  originalFile = open(originalFile, "r").read()
  testFile = open(testFile, "r").read()

  print(checkPlagiarism(originalFile, testFile))
