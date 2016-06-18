import pandas as pd
import os
import string
import spacy
import numpy as np
import array
import itertools
import json
import argparse
import time
from parse_and_split import *

def save(f, offset, rows, cols, bags, words, vals, n_words):
    assert len(rows) == len(cols) == len(bags), "Wrong list length for save"
    print offset, max(rows)+1, max(cols)+1, n_words, len(rows), len(vals)
    array.array("I", [offset, max(rows)+1, max(cols)+1, n_words, len(rows), len(vals)]).write(f)
    array.array("I", rows).write(f)
    array.array("I", cols).write(f)
    array.array("I", bags).write(f)
    array.array("I", words).write(f)
    array.array("f", vals).write(f)


def save_df(df, path, userkeys, prodkeys, offset, n_words):
    sorteddf = df.sort_values(['reviewerID', 'asin'])
    users = [userkeys[u] for u in sorteddf['reviewerID']]
    prods = [prodkeys[p] for p in sorteddf['asin']]
    bags = np.array([len(wb) for wb in sorteddf['wordBags']]).cumsum() # maybe -1?
    wbtmp = [sorted([(pos, count) for pos, count in wb.iteritems()], key=lambda x: x[0]) for wb in sorteddf['wordBags']]
    words, values = zip(*list(itertools.chain.from_iterable(wbtmp)))

    with open(path, 'wb') as f:
        save(f, offset, users, prods, bags, words, values, n_words)


def save_df_word(df, path, userkeys, prodkeys, offset, k):
    sorteddf = df.sort_values(['reviewerID', 'asin'])
    users = [userkeys[u] for u in sorteddf['reviewerID']]
    prods = [prodkeys[p] for p in sorteddf['asin']]
    for ind in range(k):
        bags = np.array([len(wb) for wb in sorteddf[ind]]).cumsum() # maybe -1?
        wbtmp = [sorted([(pos, count) for pos, count in wb.iteritems()], key=lambda x: x[0]) for wb in sorteddf[ind]]
        words, values = zip(*list(itertools.chain.from_iterable(wbtmp)))
        with open(path + `ind`, 'wb') as f:
            save(f, offset[ind], users, prods, bags, words, values, max(words)+1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", type=float, default=0.2, help="Test size")
    parser.add_argument("k", type=int, help="Split k-ways")
    parser.add_argument("data", help="Review data file")
    parser.add_argument("out", help="Output directory")

    args = parser.parse_args()

    start = time.clock()

    # build vocab
    eng = spacy.en.English()
    end = time.clock()
    print "eng", end - start

    start = time.clock()

    words_freq = get_frequent_words(get_rev(args.data, 'reviewText'), 10, eng)
    # somehow the empty string is included, mabey empty reviews?
    words_freq = [(w, count) for w, count in words_freq if w != 0]
    words, counts = zip(*words_freq)
    end = time.clock()
    print "build vocab done", end - start

    start = time.clock()

    # get word frequency per review
    bags = get_word_bags(words, get_rev(args.data, 'reviewText'), eng)

    end = time.clock()
    print "word bags done", end - start
    # parse dataframe
    df = getDF(args.data)

    # delete useless stuff
    df.drop(['reviewText', 'reviewerName', 'helpful', 'unixReviewTime', 'overall', 'reviewTime', 'summary'], axis=1, inplace=True)

    # put bags into pd df
    wbdf = pd.DataFrame({'wordBags':bags})

    # unique lists
    users = df['reviewerID'].sort_values().unique()
    prods = df['asin'].sort_values().unique()

    userkeys = {u: i for i, u in enumerate(users)}
    prodkeys = {p: i for i, p in enumerate(prods)}

    # df used to split in user and prod dimensions
    df_total = pd.concat([df,wbdf], axis=1, join='inner', copy='false')

    # save user-split train files
    for ind, (offset, df_user) in enumerate(get_k_splits(df_total, 'reviewerID', users, args.k)):
        outpath = os.path.join(args.out, "_user_train" + `ind`)
        save_df(df_user, outpath, userkeys, prodkeys, offset, len(words))
    print "saved usersplit"
    # save product-split train files
    for ind, (offset, df_prod) in enumerate(get_k_splits(df_total, 'asin', prods, args.k)):
        outpath = os.path.join(args.out, "_prod_train" + `ind`)
        save_df(df_prod, outpath, userkeys, prodkeys, offset, len(words))
    print "saved prodsplit"
    # save words-split train files
    offset, df_word = get_k_splits_words(df_total, df_total['wordBags'], words, counts, args.k)
    print df_word.columns.values
    outpath = os.path.join(args.out, "_word_train")
    save_df_word(df_word, outpath, userkeys, prodkeys, offset, args.k)
    print "saved wordsplit"

    json.dump([{'users': len(users), 'products': len(prods), 'words' : len(words)}, 
            {'vocab' : [eng.vocab.strings[w.item()] for w in words]}], 
            open(os.path.join(args.out, "meta.txt"), "w"))

if __name__ == "__main__":
    main()
