""" An example of using this library to calculate related artists
from the last.fm dataset. More details can be found
at http://www.benfrederickson.com/matrix-factorization/
This code will automically download a HDF5 version of the dataset from
GitHub when it is first run. The original dataset can also be found at
http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/lastfm-360K.html.

chcp 65001
"""
import argparse
import codecs
import logging
import time
import tqdm

import numpy as np

from implicit.als import AlternatingLeastSquares
from implicit.approximate_als import (AnnoyAlternatingLeastSquares, FaissAlternatingLeastSquares,
                                      NMSLibAlternatingLeastSquares)
from implicit.bpr import BayesianPersonalizedRanking
from implicit.nearest_neighbours import (BM25Recommender, CosineRecommender,
                                         TFIDFRecommender, bm25_weight)
from implicit.datasets.lastfm import get_lastfm

# maps command line model argument to class name
MODELS = {"als":  AlternatingLeastSquares,
          "nmslib_als": NMSLibAlternatingLeastSquares,
          "annoy_als": AnnoyAlternatingLeastSquares,
          "faiss_als": FaissAlternatingLeastSquares,
          "tfidf": TFIDFRecommender,
          "cosine": CosineRecommender,
          "bpr": BayesianPersonalizedRanking,
          "bm25": BM25Recommender}


def get_model(model_name):
    model_class = MODELS.get(model_name)
    if not model_class:
        raise ValueError("Unknown Model '%s'" % model_name)

    # some default params
    if issubclass(model_class, AlternatingLeastSquares):
        params = {'factors': 64, 'dtype': np.float32}
    elif model_name == "bm25":
        params = {'K1': 100, 'B': 0.5}
    elif model_name == "bpr":
        params = {'factors': 63}
    else:
        params = {}

    return model_class(**params)


def calculate_similar_artists(output_filename, model_name="als"):
    """ generates a list of similar artists in lastfm by utiliizing the 'similar_items'
    api of the models """
    artists, users, plays = get_lastfm()

    # create a model from the input data
    model = get_model(model_name)

    # if we're training an ALS based model, weight input for last.fm
    # by bm25
    if issubclass(model.__class__, AlternatingLeastSquares):
        # lets weight these models by bm25weight.
        logging.debug("weighting matrix by bm25_weight")
        plays = bm25_weight(plays, K1=100, B=0.8)

        # also disable building approximate recommend index
        model.approximate_recommend = False

    # this is actually disturbingly expensive:
    plays = plays.tocsr()

    logging.debug("training model %s", model_name)
    start = time.time()
    model.fit(plays)
    logging.debug("trained model '%s' in %0.2fs", model_name, time.time() - start)

    # write out similar artists by popularity
    start = time.time()
    logging.debug("calculating top artists")

    user_count = np.ediff1d(plays.indptr)
    to_generate = sorted(np.arange(len(artists)), key=lambda x: -user_count[x])

    # write out as a TSV of artistid, otherartistid, score
    logging.debug("writing similar items")
    with tqdm.tqdm(total=len(to_generate)) as progress:
        with codecs.open(output_filename, "w", "utf8") as o:
            for artistid in to_generate:
                artist = artists[artistid]
                for other, score in model.similar_items(artistid, 11):
                    o.write("%s\t%s\t%s\n" % (artist, artists[other], score))
                progress.update(1)

    logging.debug("generated similar artists in %0.2fs",  time.time() - start)


def calculate_recommendations(output_filename, model_name="als"):
    """ Generates artist recommendations for each user in the dataset """
    # train the model based off input params
    artists, users, plays = get_lastfm()

    # for i in range(len(users)):
    #     print(users[i], end=' ')
    #     for j in range(len(artists)):
    #         if plays[i, j]!=0:
    #             print(plays[i, j], end=' ')
    #     print()

    print(type(users),users.shape)
    print(type(artists),artists.shape)
    print(type(plays),plays.shape)
    return

    # create a model from the input data
    model = get_model(model_name)

    # if we're training an ALS based model, weight input for last.fm
    # by bm25
    if issubclass(model.__class__, AlternatingLeastSquares):
        # lets weight these models by bm25weight.
        logging.debug("weighting matrix by bm25_weight")
        plays = bm25_weight(plays, K1=100, B=0.8)

        # also disable building approximate recommend index
        model.approximate_similar_items = False

    # this is actually disturbingly expensive:
    plays = plays.tocsr()

    logging.debug("training model %s", model_name)
    start = time.time()
    model.fit(plays)
    logging.debug("trained model '%s' in %0.2fs", model_name, time.time() - start)

    # generate recommendations for each user and write out to a file
    start = time.time()
    user_plays = plays.T.tocsr()
    with tqdm.tqdm(total=len(users)) as progress:
        with codecs.open(output_filename, "w", "utf8") as o:
            for userid, username in enumerate(users):
                for artistid, score in model.recommend(userid, user_plays):
                    o.write("%s\t%s\t%s\n" % (username, artists[artistid], score))
                progress.update(1)
    logging.debug("generated recommendations in %0.2fs",  time.time() - start)

if __name__ == "__main__":
    calculate_recommendations('./test.txt', model_name='als')
