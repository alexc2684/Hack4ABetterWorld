import csv
import pandas as pd
import numpy as np
import ArgumentParser
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess(sent):
    string = ''
    for word in sent.split(' '):
        word = word.lower()
        #clean text outside of stop words
        if word.find('https') == -1 and word.find('amp') == -1:
            curr = ps.stem(word)
            string += curr + ' '
    return string

def bag_of_words(keyword, n):
    rows = db.loc[db['u4u_dataset'] == keyword]
    text = rows['full_text'].values
    count_vec = CountVectorizer(preprocessor=preprocess, stop_words=stop_words)
    vec = count_vec.fit(text)
    bag_of_words = vec.transform(text)

    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]

if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-f', dest='path')
  parser.add_argument('-n', dest='num_words')
  args = parser.parse_args()

  db = pd.read_csv(args.path, lineterminator='\n')
  keywords = db['u4u_dataset'].value_counts()

  common_words = {}
  for keyword in keywords.index:
      common_words[keyword] = bag_of_words(keyword, args.num_words)
