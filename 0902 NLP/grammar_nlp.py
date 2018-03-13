import random

grammar = {
    "_S" : ["_NP _VP"],
    "_NP": ["_N", "_A _NP _P _A _N"],
    "_VP": ["_V", "_V _NP"],
    "_N" : ["data science", "Python", "regression"],
    "_A" : ["big", "linear", "logistic"],
    "_P" : ["about", "near"],
    "_V" : ["learns", "test", "is"]
}
#_있으면 확장 가능한 규칙 rule
# otherwise, terminal 종결어
#_S: sentence, _NP: noun phrase ...

def is_terminal(token):
    return token[0] != "_"
#return 0 false if there is  _
#return 1 true if it is terminal

#print(is_terminal("_NP"))

def expand(grammar, tokens):
    for i, token in enumerate(tokens):

        # ignore terminals
        if is_terminal(token):
            continue

        # choose a replacement at random
        replacement = random.choice(grammar[token])
        #print("repl ", replacement)
        print("token: ", token)
        print("gram: ", grammar[token])
        print("repl: ", replacement)

        if is_terminal(replacement):
            tokens[i] = replacement
        else:
            #randomly chosen token is not terminal
            print("before: ", tokens)
            tokens = tokens[:i] + replacement.split() +tokens[(i+1):]
            print("after: ", tokens)
        return expand(grammar, tokens)

    # if we get here we had all terminals and are done
    return tokens

def generate_sentence(grammar):
    return expand(grammar, ["_S"])

generate_sentence(grammar)