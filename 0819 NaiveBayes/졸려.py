import glob, re, math, random
from collections import Counter, defaultdict

def get_data(path):
    data = []
    for fn in glob.glob(path):
        is_spam = "ham" not in fn
        with open(fn,'r') as file:
            try:
                lines = file.readlines()
            except UnicodeDecodeError as e: continue
            else:
                for line in lines:
                    if line.startswith("Subject:"):
                       subject=re.sub(r'^Subject:',"",line).strip()
                       data.append((subject, is_spam))
    return data

def split_data(data, prob):
     results = [], []
     for row in data:
        results[0 if random.random() < prob else 1].append(row)
     return results

def tokenize(message):
    message = message.lower()                       # convert to lowercase
    all_words = re.findall("[a-z0-9']+", message)   # extract the words
    return set(all_words)                           # remove duplicates

def count_words(training_set):
    """training set consists of pairs (message, is_spam)"""
    counts = defaultdict(lambda: [0, 0])
    for message, is_spam in training_set:
        for word in tokenize(message):
            counts[word][0 if is_spam else 1] += 1
    return counts

def div_mail(training_set):
    num_spams = len([is_spam
                     for message, is_spam in training_set
                     if is_spam])
    num_non_spams = len(training_set) - num_spams
    return num_spams , num_non_spams

def word_probabilities(counts, total_spams, total_non_spams):
    """turn the word_counts into a list of triplets
    w, p(w | spam) and p(w | ~spam)"""
   return [(w,
            (spam + k) / (total_spams + 2 * k),
            (non_spam + k) / (total_non_spams + 2 * k))
            for w, (spam, non_spam) in counts.items()]


def spam_probability(word_probs, message):
    message_words = tokenize(message)
    log_prob_if_spam = log_prob_if_not_spam = 0.0

    for word, prob_if_spam, prob_if_not_spam in word_probs:

        if word in message_words:
            log_prob_if_spam += math.log(prob_if_spam)
            log_prob_if_not_spam += math.log(prob_if_not_spam)
        else:
            log_prob_if_spam += math.log(1.0 - prob_if_spam)
            log_prob_if_not_spam += math.log(1.0 - prob_if_not_spam)

    prob_if_spam = math.exp(log_prob_if_spam)
    prob_if_not_spam = math.exp(log_prob_if_not_spam)
    return prob_if_spam / (prob_if_spam + prob_if_not_spam)

class NaiveBayesClassifier:

    def __init__(self,path, k=0.5):
        self.k = k
        self.word_probs = []
        self.data = get_subject_data(path)
        random.seed(0)
        self.train_data, self.test_data = split_data(self.data, 0.75)

    def train(self, train_data):
        # count spam and non-spam messages
        num_spams, num_non_spams = div_mail(train_data)
        # run training data through our "pipeline"
        word_counts = count_words(train_data)
        self.word_probs = word_probabilities(word_counts,
                                            num_spams,
                                            num_non_spams,
                                            self.k)

    def classify(self,message):
        return spam_probability(self.word_probs, message)

a =NaiveBayesClassifier(path)
a.train(a.train_data)
classified = [(subject, is_spam, a.classify(subject))
              for subject, is_spam in a.test_data]
print(classified)

counts = Counter((is_spam, spam_probability > 0.5)
                     for _, is_spam, spam_probability in classified)
print (counts)
