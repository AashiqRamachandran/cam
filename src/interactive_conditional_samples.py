#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf
import tweepy
import time
from textblob import TextBlob
import csv
import re

import model, sample, encoder

#twitter consumer tokens being defined here
#IUK4LzRRBjNZSPse0Yrh0ASzs
#R7wgP61StMDNKtSN71hWTj7ROjIa15m8643bfm9Cy8XsUG6BSR
auth = tweepy.OAuthHandler("SGMkyomaXPvKM3B6fLyuTw3Dn", "aIa7YzrL3znFDFuVOB9E9wCFoQM873VHDYrqU9uPVZFm34HvwO")
#twitter access tokens being defined here
#1226323598144458754-gAWJFxK7fzoioaPomxaUS7PemHhDIl
#MtMZOPHtFyTmAu14zwRRrgJNUZx8gDZUuItwXIz8VFMHv
auth.set_access_token("1226323598144458754-ap3KOIhezEXRj6wdw2L6cvdEw8N8Ij", "wWOhdDNarsrOkaayQHC95WGipvHf5axSTSu9jwGHjS8Po")
#twitter api call being set here
api = tweepy.API(auth, wait_on_rate_limit=True)

global reply
global tweet_data

def postt(message, idom):
    api.update_status(message, idom)
    time.sleep(2)

def twitter_search(keywords):
    search = tweepy.Cursor(api.search, q=keywords, result_type="recent", lang="en").items(10)
    with open("data.csv","w") as file:
        writer = csv.writer(file)    
        writer.writerow(["user id", "tweeted text", "replied text", "sentiment polarity", "sentiment objectivity"])
        for item in search:
            #tweet_data=item.text
            screen_name=item.user.screen_name
            reply=fire.Fire(interact_model(item.text))
            final_reply = re.sub(r"http\S+", "", reply)
            link=" Check out https://bit.ly/2UcUNrp"
            message="@%s I think " %(screen_name) + str(final_reply)+ str(link)
            postt(message, item.id)
            sentiment_overall=TextBlob(item.text)            
            print(sentiment_overall.sentiment)
            writer.writerow([item.user.screen_name, item.text, message, sentiment_overall.sentiment.polarity, sentiment_overall.sentiment.subjectivity])
    print("End of code")
        
def interact_model(
    tweet_data,
    model_name='124M',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=40,
    temperature=1,
    top_k=40,
    top_p=1,
    models_dir='models',
):
    """
    Interactively run the model
    :model_name=124M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=40 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
     :models_dir : path to parent folder containing model subfolders
     (i.e. contains the <model_name> folder)
    """
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        while True:
            raw_text = tweet_data
            context_tokens = enc.encode(raw_text)
            generated = 0
            for _ in range(nsamples // batch_size):
                out = sess.run(output, feed_dict={
                    context: [context_tokens for _ in range(batch_size)]
                })[:, len(context_tokens):]
                for i in range(batch_size):
                    generated += 1
                    text = enc.decode(out[i])
                    print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                    return text
                    print(text)
            print("=" * 80)

if __name__ == '__main__':
    keywords = str(input('Enter terms to search in quotes: '))
    keywords = [keyword.strip() for keyword in keywords.split(',')]
    twitter_search(keywords)
    #fire.Fire(interact_model)

