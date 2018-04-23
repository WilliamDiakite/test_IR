import os

from utils import tokenize, save_obj, load_obj
from itertools import chain
from collections import Counter
from collections import defaultdict


# import some stopwords
with open('stopwords.txt') as f:
    stopwords = [s.rstrip() for s in f]


# import the documents and their annotations
documents = []
annotations = []

for f in os.listdir('./data/NLM_500/documents/'):
    filename = './data/NLM_500/documents/' + f
    if filename.endswith('.txt'):
        documents.append(open(filename, encoding='ISO-8859-1').read())
    elif filename.endswith('.key'):
        an = [a.rstrip().lower() for a in open(filename,
                                               encoding='ISO-8859-1')]
        annotations.append(an)

# print(documents[0])
# print()
# print(annotations[0])

# tokenize documents
print('[ + ] Tokenizing documents')
documents = [tokenize(d, stopwords) for d in documents]
documents = [list(set(d)) for d in documents]


# Output some infos about the data
vocab = []
thesaurus = []

for doc in documents:
    vocab += list(set(doc))

for tw in annotations:
    thesaurus += tw

vocab = sorted(list(set(vocab)))
thesaurus = sorted(list(set(thesaurus)))
intersection = sorted(list(set(thesaurus).intersection(vocab)))

print('Vocab size:', len(vocab))
print('Thesaurus size:', len(thesaurus))
print('Intersection size:', len(intersection))


# Tag quantity (and not really a distribution)
nb_tags = 0
min_doc = 5
tag_dist = Counter(chain.from_iterable(annotations))
for t in sorted(tag_dist.items(), key=lambda x: x[1], reverse=True):
    if t[1] >= min_doc:
        nb_tags += 1
print('[ ! ]Tags that tag more than {} documents: {}'.format(min_doc, nb_tags))

# Init training
# TODO: cross validation

# Get training test
documents_train = documents[:375]
annotations_train = annotations[:375]

# Get test set
documents_test = documents[375:]
annotations_test = annotations[375:]


def word_occurence(documents, vocab):
    # Init a zero vector of size vocab
    word_count_idx = dict((w, 0) for i, w in enumerate(vocab))

    for doc in documents:
        # Update vectors for all the collection
        for word in doc:
            word_count_idx[word] += 1

    return word_count_idx


def compute_mle_vector(word_occurence, N):
    mle_vector = dict()
    for w in word_occurence:
        mle_vector[w] = (word_occurence[w] + 0.5) / (N + 1)

    return mle_vector


def compute_relevance_matrices(documents, annotations, thesaurus, vocab):

    for th in thesaurus:
        with open('last_th.txt', 'w') as f:
            f.write(th)

        # hepls separate relevant docs from non-relevant ones
        corpus = defaultdict(lambda: [])

        # mark relevance for all documents
        for doc, tags, i in zip(documents, annotations, range(len(documents))):
            if th in tags:
                corpus['relevant'].append(doc)
            else:
                corpus['nonrelevant'].append(doc)

        # Word occurrences in relevant and non relevant documents
        rel_count_vec = word_occurence(corpus['relevant'], vocab)
        non_count_vec = word_occurence(corpus['nonrelevant'], vocab)

        # Number of relevant and non-relevant documents
        N_rel = len(corpus['relevant'])
        N_non = len(corpus['nonrelevant'])

        # Compute maximum likelihood
        p_prob = compute_mle_vector(rel_count_vec, N_rel)
        q_prob = compute_mle_vector(non_count_vec, N_non)

        # save probabilities on disk
        save_obj(obj=(p_prob, q_prob), name=th)


def score(p_vec, q_vec, new_doc):
    '''
        For each word of the thesaurus, we compute
        the probability of new_doc of being relevant
    '''
    num_prod = 1
    denum_prod = 1
    score = 0

    for t in new_doc:
        num_prod *= p_vec[t] * (1 - q_vec[t])
        denum_prod *= q_vec[t] * (1 - p_vec[t])

    try:
        score = num_prod / denum_prod
    except Exception as e:
        pass

    return score


def predict_tags(thesaurus, new_doc, n_best=10):
    scores = dict()

    for th in thesaurus:
        p_vec, q_vec = load_obj(th)
        scores[th] = score(p_vec, q_vec, new_doc)

    scores = sorted(scores.items(), key=lambda x: x[1])

    return scores[:n_best]


def test():
    # define some shit data
    docs = []
    doc1 = ['i', 'love', 'paris']
    doc2 = ['i', 'love', 'cats']
    doc3 = ['am', 'allergic', 'cats']

    docs.append(doc1)
    docs.append(doc2)
    docs.append(doc3)

    voc = doc1 + doc2 + doc3
    voc = list(set(voc))
    print('vocab:', voc)

    # Here thesaurus and annotations are the same
    thes = ['animals', 'city', 'health']
    anno = [['city'], ['animals'], ['health']]

    # check relevance matrix
    compute_relevance_matrices(docs, anno, thes, voc)
    for t in thes:
        print('Thesaurus word:', t)
        t_p, t_q = load_obj(t)
        for w in voc:
            print('word:', w, 't_p:', t_p[w], 't_q:', t_q[w])
        print()

    doc4 = ['cats', 'love', 'am']
    print(doc4)

    doc4_scores = predict_tags(thes, doc4)
    for th_s in doc4_scores:
        print('thesaurus:', th_s[0], 'relevance:', th_s[1])


print('\n========== TEST ============\n')
test()

print('\n========== TRAIN ============\n')
# compute_relevance_matrices(documents_train, annotations_train,
#                            thesaurus, vocab)
print('[ + ] finished training')


print('\n========== SCORE ============\n')
print('[...] Computing document relevance')
results = []
for doc, tags in zip(documents_test, annotations_test):
    predicted = predict_tags(thesaurus, doc, n_best=len(tags))
    results.append((predicted, tags))

print('[...] Computing model accuracy')
accuracy = 0
for r in results:
    accuracy += len(set(r[0]).intersection(set(r[1]))) / len(results)

print('[ + ] Model accuracy: {} %'.format(accuracy))
