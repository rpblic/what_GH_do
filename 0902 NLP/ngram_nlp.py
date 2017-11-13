from __future__ import division
import math, random, re
from collections import defaultdict, Counter
from bs4 import BeautifulSoup
import requests

# n-gram models
#
def fix_unicode(text):
    return text.replace(u"\u2019", "'")


def get_document():

    url = "https://www.oreilly.com/ideas/what-is-data-science"
    html = requests.get(url).text
    #print(html)
    soup = BeautifulSoup(html, 'html.parser')
    #html5lib or html.parser

    content = soup.find("div", "article-body")        # find article-body div
    regex = r"[\w']+|[\.]"                            # matches a word or a period

    document = []

    for paragraph in content("p"):
        words = re.findall(regex, fix_unicode(paragraph.text))
        #print(words)
        document.extend(words)
    #print(document)
    return document

newline = [dot.replace('.', '.\n') for dot in get_document()]
sentence = ' '.join(newline)
#print(sentence)

document = get_document()
bigrams = zip(document, document[1:])
transitions= defaultdict(list)
for prev, current in bigrams:
    transitions[prev].append(current)
    #print(transitions)


def generate_using_bigrams(transitions):
    current = "."   # this means the next word will start a sentence
    result = []
    while True:
        next_word_candidates = transitions[current]    # bigrams (current, _)
        print("tran: ", transitions[current])
        current = random.choice(next_word_candidates)  # choose one at random
        result.append(current)                         # append it to results
        if current == ".":
            return " ".join(result)     # if "." we're done

print(generate_using_bigrams(transitions))

trigrams = zip(document, document[1:], document[2:])
trigram_transitions= defaultdict(list)
starts=[]
for prev, current, next in trigrams:
    if prev==".":
        starts.append(current)
    trigram_transitions[(prev, current)].append(next)
    #print(trigram_transitions[(prev, current)])

def generate_using_trigrams(starts, trigram_transitions):
    current = random.choice(starts)   # choose a random starting word
    prev = "."                        # and precede it with a '.'
    result = [current]
    while True:
        next_word_candidates = trigram_transitions[(prev, current)]
        next = random.choice(next_word_candidates)

        prev, current = current, next
        result.append(current)

        if current == ".":
            return " ".join(result)

#print(generate_using_trigrams(starts, trigram_transitions))