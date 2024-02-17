import json
import string
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict
# Regex Operations
import re
# Vektorisierung mit Bag-of-Words
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd # Anzeige des BoW mit Pandas 
# Vektorisierung mit Tfidf
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
#LDA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
#LSA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


##Bereinigung des Textes
# Lade den JSON-Datensatz
with open('Ordnungsamt-Onlinemeldungen.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# index Daten in Text to clean speichern
text_to_clean = data['index']

# Ausgabe zur Überprüfung des importierten Textes
#print(text_to_clean)

# Erstellen der beiden Arrays Betreff und Sachverhalt
betreff_liste = []
sachverhalt_liste = []

# Extrahiere 'betreff' und 'sachverhalt' aus jedem Eintrag
for eintrag in text_to_clean:
    if 'betreff' in eintrag:
        betreff_liste.append(eintrag['betreff'])
    if 'sachverhalt' in eintrag:
        sachverhalt_liste.append(eintrag['sachverhalt'])
"""
# Ausgabe der extrahierten Listen
print("Betreff-Liste:")
print(betreff_liste)
print("Sachverhalt-Liste:")
print(sachverhalt_liste)
"""

def remove_unwanted_characters(token):
    # Entfernen von Punkten und anderen speziellen Zeichen, außer Buchstaben (einschließlich Umlauten)
    token = re.sub(r'[^a-zA-ZäöüÄÖÜß\s]', '', token)
    return token

# Funktion zur Bereinigung des Textes
def preprocess_complex_words(text):
    # Ersetze spezifische Muster durch angepasste Formen
    text = re.sub(r'Park- und Haltverbot', 'Park-/Haltverbot', text)
    text = re.sub(r'Anwohner-, Gästevignetten', 'Anwohner-/Gästevignetten', text)
    return text

def clean_text(text):
    if text is None:
        return []
    
    text = preprocess_complex_words(text)
    tokens = word_tokenize(text)

    # Entfernen von Zahlen
    tokens = [token for token in tokens if not re.match(r'\d+\.?\d*', token)]

    # Entfernen von Punkten, speziellen Zeichen und Emojis
    tokens = [remove_unwanted_characters(token) for token in tokens]

    # Liste der zu entfernenden spezifischen Wörter
    unwanted_words = {'ca', 'w3w5xd', 'seit'} 
    tokens = [token for token in tokens if token.lower() not in unwanted_words]

    # Entfernung von Stoppwörtern
    stop_words = set(stopwords.words('german')) 
    tokens = [word for word in tokens if word.lower() not in stop_words and word not in string.punctuation]
    
    # Lemmatisierung
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Umwandlung in Kleinbuchstaben
    tokens = [word.lower() for word in tokens]
    return tokens
# Bereinigung für jeden Betreff
# None einträge rausfiltern
cleaned_betreff_liste = [clean_text(betreff) for betreff in betreff_liste if betreff is not None]

# Ausgabe der bereinigten Betreff-Liste
#for index, betreff in enumerate(cleaned_betreff_liste[:5]): # Fall nur die ersten 5 relevant sind
#    print(f'{index + 1}. {betreff}')
print("Betreff-Liste (erste und letzte 5 Einträge):")
if len(cleaned_betreff_liste) > 10:
    for index, betreff in enumerate(cleaned_betreff_liste[:5]):
        print(f'{index + 1}. {betreff}')
    print("...")
    for index, betreff in enumerate(cleaned_betreff_liste[-5:], start=len(cleaned_betreff_liste)-5):
        print(f'{index + 1}. {betreff}')
else:
    for index, betreff in enumerate(cleaned_betreff_liste):
        print(f'{index + 1}. {betreff}')

# Bereinigung für jeden Sachverhalt
# None einträge rausfiltern
cleaned_sachverhalt_liste = [clean_text(sachverhalt) for sachverhalt in sachverhalt_liste if sachverhalt is not None]

# Ausgabe der bereinigten Sachverhalts-Liste
#for index, sachverhalt in enumerate(cleaned_sachverhalt_liste[:5]): # Fall nur die ersten 5 relevant sind
#    print(f'{index + 1}. {sachverhalt}')

print("Sachverhalt-Liste (erste und letzte 5 Einträge):")
if len(cleaned_sachverhalt_liste) > 10:
    for index, sachverhalt in enumerate(cleaned_sachverhalt_liste[:5]):
        print(f'{index + 1}. {sachverhalt}')
    print("...")
    for index, sachverhalt in enumerate(cleaned_sachverhalt_liste[-5:], start=len(cleaned_sachverhalt_liste)-5):
        print(f'{index + 1}. {sachverhalt}')
else:
    for index, sachverhalt in enumerate(cleaned_sachverhalt_liste):
        print(f'{index + 1}. {sachverhalt}')


## BoW mit Betreff & Sachverhalte
# Bag of Words Funktion
def generate_bag_of_words(text_list):
    # Daten vorbereiten > in Strings umwandeln
    text_strings = [' '.join(tokens) for tokens in text_list]

    # Erstellen einer Instanz von CountVectorizer; CountVectorizer erwartet einen String
    vectorizer = CountVectorizer(ngram_range=(1,2))

    # Vektorisierer auf Daten anwenden
    text_bow = vectorizer.fit_transform(text_strings)

    # Ermitteln der Worthäufigkeit
    word_freq = dict(zip(vectorizer.get_feature_names_out(), np.asarray(text_bow.sum(axis=0)).ravel()))
    word_freq_sorted = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True))

    return word_freq_sorted
    
# Anwenden der Funktion auf die Betreff_liste & sachverhalt_liste
word_freq_betreff = generate_bag_of_words(cleaned_betreff_liste)
word_freq_sachverhalt = generate_bag_of_words(cleaned_sachverhalt_liste)
"""
Für die komlette Ausgabe der Betreff- oder Sachverhalt-Liste:
# Ausgeben BoW Betreff-Liste
print("Betreff-Liste:")
print(word_freq_betreff)

# Ausgeben BoW Sachverhalt Liste
print("Sachverhalt-Liste:")
print(word_freq_sachverhalt)
"""
# Funktion DataFrame mit der Worthäufigkeit erstellen
def bow_dataframe(word_freq_text):
    df = pd.DataFrame(list(word_freq_text.items()), columns=['Wort', 'Häufigkeit'])
    # Sortierte Liste mit top 5 Wörter
    top_5_words = df.sort_values(by='Häufigkeit', ascending=False).head(5)
    return top_5_words

# Worthäufigkeit Betreff
df_betreff = bow_dataframe(word_freq_betreff)
print("Betreff-Worthäufigkeit:")
print(df_betreff)

# Worthäufigkeit Sachverhalt
df_sachverhalt = bow_dataframe(word_freq_sachverhalt)
print("Sachverhalt-Worthäufigkeit:")
print(df_sachverhalt)

## TF-IDF mit Betreff & Sachverhalte - Alle Einträge als ein einzelnes Dokument
def tfidf_vectorize_as_single_document(token_lists):
    # Umwandlung der Token-Listen in Strings und Zusammenführen zu einem einzigen Dokument
    combined_text = ' '.join([' '.join(tokens) for tokens in token_lists])

    # Erstellung und Anwendung des TfidfVectorizer
    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    tfidf_matrix = vectorizer.fit_transform([combined_text])

    # Umwandlung der TF-IDF-Matrix in einen DataFrame
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    return tfidf_df

# Anwendung der TF-IDF-Vektorisierungsfunktion direkt auf die Token-Listen - Betreff
betreff_tfidf_df = tfidf_vectorize_as_single_document(cleaned_betreff_liste)
# Anwendung der TF-IDF-Vektorisierungsfunktion direkt auf die Token-Listen - Sachverhalt
sachverhalt_tfidf_df = tfidf_vectorize_as_single_document(cleaned_sachverhalt_liste)

"""
# Ausgabe der gesamten Listen
print("Betreff-Worthäufigkeit:")
print(betreff_tfidf_df)
print("Sachverhalt-Worthäufigkeit:")
print(sachverhalt_tfidf_df)
"""

def transform_and_sort_tfidf(tfidf_df):
    # Transformation des DataFrames
    transformed_df = tfidf_df.T.reset_index()
    transformed_df.columns = ['Wort', 'Wert']

    # Sortierung des DataFrames in absteigender Reihenfolge nach 'Wert'
    sorted_df = transformed_df.sort_values(by='Wert', ascending=False)

    return sorted_df

# Anwenden der Funktion auf den TF-IDF DataFrame für Betreff
sorted_betreff_tfidf = transform_and_sort_tfidf(betreff_tfidf_df)
print("Betreff-Tf-idf:")
print(sorted_betreff_tfidf.head())  

# Anwenden der Funktion auf den TF-IDF DataFrame für Sachverhalt
sorted_sachverhalt_tfidf = transform_and_sort_tfidf(sachverhalt_tfidf_df)
print("Sachverhalt-Tf-idf:")
print(sorted_sachverhalt_tfidf.head()) 


## LDA für Betreff & Sachverhalt
def perform_lda(text_list, n_topics=5, n_top_words=10):
    # Erstellen und Anwenden des TfidfVectorizer
    vectorizer = TfidfVectorizer()
    text_tfidf = vectorizer.fit_transform(text_list)

    # Erstellen und Anwenden der LDA
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=0)
    lda.fit(text_tfidf)

    # Anzeigen der Top-Wörter für jedes Thema
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        print(f"Thema {topic_idx + 1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print()

betreff_strings = [' '.join(tokens) for tokens in cleaned_betreff_liste]
sachverhalt_strings = [' '.join(tokens) for tokens in cleaned_sachverhalt_liste]

# Anwendung der LDA-Funktion auf die Betreff- und Sachverhalt-Daten
print("LDA für Betreff:")
perform_lda(betreff_strings)

print("LDA für Sachverhalt:")
perform_lda(sachverhalt_strings)

## LSA für Betreff & Sachverhalt
def perform_lsa(text_list, n_topics=5, n_top_words=10):
    # Erstellung und Anwendung des TfidfVectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(text_list)

    # Durchführung der LSA
    svd = TruncatedSVD(n_components=n_topics)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    lsa_matrix = lsa.fit_transform(tfidf_matrix)

    # Anzeigen der Top-Wörter für jedes Thema
    feature_names = vectorizer.get_feature_names_out()
    for i, comp in enumerate(svd.components_):
        terms_comp = zip(feature_names, comp)
        sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:n_top_words]
        print(f"Thema {i+1}:")
        print(" ".join([t[0] for t in sorted_terms]))
        print()

    return lsa_matrix

betreff_strings = [' '.join(tokens) for tokens in cleaned_betreff_liste]
sachverhalt_strings = [' '.join(tokens) for tokens in cleaned_sachverhalt_liste]

# LSA durchführen für Betreff
print("LSA für Betreff:")
betreff_lsa_matrix = perform_lsa(betreff_strings, n_topics=5, n_top_words=10)

# LSA durchführen für Sachverhalt
print("LSA für Sachverhalt:")
sachverhalt_lsa_matrix = perform_lsa(sachverhalt_strings, n_topics=5, n_top_words=10)
