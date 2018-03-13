from __future__ import division
import math, random, re
from collections import defaultdict, Counter
from bs4 import BeautifulSoup
import requests

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
        # a = [1,2] a.extend([3,4]) assert a == [1, 2, 3, 4]
        # a = [1,2] a.append([3,4]) assert a == [1, 2, [3, 4]]
        # not [1, 2, 3, 4]

    #print(document)
    return document

def generate_using_bigrams(transitions):
    current= "."        # this means the next word will start a sentence
    result=[]
    while True:
        next_word_candidates= transitions[current]      #bigrams (current, _)
        current= random.choice(next_word_candidates)    # choose one at random
        result.append(current)                          # append it to results
        if current== '.': return " ".join(result)       # if "." we're done

def expand(grammar, tokens):
    for i, token in enumerate(tokens):
        if is_terminal(token): continue                 # ignore terminals
        replacement= random.choice(grammar[token])      # choose a replacement at random
        if is_terminal(replacement):
            tokens[i]= replacement
        else:
`            tokens= tokens[:i]+ replacement.split()+ tokens[(i+1):]
    `   return expand(grammar, tokens)
    return tokens                                       # if we get here we had all terminals and are done

def generate_sentence(grammar):
    return expand(grammar, ["_s"])

document = get_document()
bigrams = zip(document, document[1:])
transitions= defaultdict(list)
grammar = {"_S" : ["_NP _VP"],\
"_NP": ["_N", "_A _NP _P _A _N"],\
"_VP": ["_V", "_V _NP"],\
"_N" : ["data science", "Python", "regression"],\
"_A" : ["big", "linear", "logistic"],\
 "_P" : ["about", "near"],\
"_V" : ["learns", "trains", "test", "is"]}
#언더바 표시 _ 있으면 확장 가능한 규칙 rule
#나머지는 종결어 terminal
#_S: sentence, _NP: noun phrase ...
for prev, current in bigrams: transitions[prev].append(current)

print(generate_using_bigrams(transitions))
