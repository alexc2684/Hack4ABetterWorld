from nltk.stem import PorterStemmer
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

def display_topics(model, feature_names, no_top_words):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        topics.append([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]])
    return topics
