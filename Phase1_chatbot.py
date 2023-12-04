import nltk
from nltk.tokenize import word_tokenize
from nltk.chunk import RegexpParser
import string

# Function to replace punctuation with space
def replace_punctuation_with_space(text):
    return "".join([char if char not in string.punctuation else ' ' for char in text])

# Function for text normalization (lowercasing)
def normalize_text(text):
    return text.lower()

# Function for tokenization
def tokenize_text(text):
    return word_tokenize(text)

# Function to remove stopwords
def remove_stopwords(text):
    stopwords = set(nltk.corpus.stopwords.words('english'))
    return [word for word in text if word.lower() not in stopwords]

# Function to perform part-of-speech tagging
def pos_tagging(text):
    return nltk.pos_tag(text)

# Function to extract adjective/CD-noun or hyphenated noun pairs and noun-noun pairs
def extract_noun_pairs(tagged_text):
    grammar = r'NP: {<JJ.*|CD>?<NN.*|NNS|VB.*>-<NN.*|NNS|VB.*>|<JJ.*|CD>?<NN.*|NNS|VB.*>|<NN.*|NNS|VB.*>-<NN.*|NNS|VB.*>|<CD><NN.*|NNS|VB.*>|<NN.*|NNS|VB.*><NN.*|NNS|VB.*>}'
    chunk_parser = RegexpParser(grammar)
    tree = chunk_parser.parse(tagged_text)

    pairs = []

    for subtree in tree.subtrees():
        if subtree.label() == 'NP':
            words = [word for word, pos in subtree.leaves()]
            adj_cd = [word for word, pos in subtree.leaves() if pos in ['JJ', 'JJR', 'JJS', 'CD', 'VB'] and word != subtree]
            noun = ' '.join(word for word, pos in subtree.leaves() if pos in ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])
            
            if noun:
                if adj_cd:
                    pairs.append((adj_cd, noun))
                else:
                    pairs.append((None, noun))

    return pairs

# Function to perform the entire process
def process_user_intent(user_intent):
    # Step 1: Replace punctuation with space
    cleaned_text = replace_punctuation_with_space(user_intent)

    # Step 2: Normalize text (lowercasing)
    normalized_text = normalize_text(cleaned_text)

    # Step 3: Tokenize text
    tokens = tokenize_text(normalized_text)

    # Step 4: Remove stopwords
    filtered_tokens = remove_stopwords(tokens)

    # Step 5: Perform part-of-speech tagging
    tagged_text = pos_tagging(filtered_tokens)

    # Step 6: Extract adjective/CD-noun or hyphenated noun pairs and noun-noun pairs
    noun_pairs = extract_noun_pairs(tagged_text)

    return noun_pairs

# # Example usage:

# user_intent = "The chatbot should suggest a diet plan that is not only low in fat but also tailored to help manage high cholesterol levels."

# # Process the user intent
# result_pairs = process_user_intent(user_intent)

# # Print the result
# for pair in result_pairs:
#     print(pair)
