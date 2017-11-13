#-*- coding: utf-8 -*-

import tweepy, api
import sys

consumer_key= "igB5bYrsyt5XyX3wBq9TTmCLM"
consumer_skey= "5fkhbYvVE6cJDuIJTKuidYv8la9i4UkDs8a5rUOCChhPpIhkHu"
access_token= "466142961-UKC102IQ1XxyyQtXBdTwyVqCh1nfRkug1xFaAjU5"
access_stoken= "bfgNcqT8TywcQ6vlXXCcDSs6xisUvFTZaYAkPDmD0Imb3"

auth= tweepy.OAuthHandler(consumer_key, consumer_skey)
auth.set_access_token(access_token, access_stoken)
api= tweepy.API(auth)

dataLength= 0

class MyStreamListener(tweepy.StreamListener):
    def on_status(self, status):
        print(status.text)

    def on_error(self, status_code):
        if status_code== 420:
            return False


if __name__== '__main__':
    mySL= MyStreamListener()
    myStream= tweepy.Stream(auth= api.auth, listener= mySL)
    myStream.filter(track= ["movie"])
