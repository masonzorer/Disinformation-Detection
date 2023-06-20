import datetime
import os
import tweepy
import json
import pandas as pd
from dotenv import load_dotenv
from typing import List
import database
import pymongo

def query_twitter(query, collection_name, start_date, end_date):

    client = get_client()
    start_date_arr = start_date.split()
    end_date_arr = end_date.split()
    get_timed_tweets_from_api_into_database(client, query, datetime.datetime(int(start_date_arr[0]), int(start_date_arr[1]), int(start_date_arr[2]), 0, 0, 0, tzinfo=datetime.timezone.utc), datetime.datetime(int(end_date_arr[0]), int(end_date_arr[1]), int(end_date_arr[2]), 1, 0, 0, tzinfo=datetime.timezone.utc), collection_name)

def get_client():

    file_exists = os.path.exists('.env')
    if file_exists:
        load_dotenv()
    else:
        load_dotenv(dotenv_path = Path('../.env'))

    bearer_token = os.getenv("BEARER_TOKEN")
    consumer_key = os.getenv("API_KEY")
    consumer_secret = os.getenv("API_KEY_SECRET")
    access_token = os.getenv("ACCESS_TOKEN")
    access_token_secret = os.getenv("ACCESS_TOKEN_SECRET")

    client = tweepy.Client(bearer_token=bearer_token,
                           consumer_key=consumer_key,
                           consumer_secret=consumer_secret,
                           access_token=access_token,
                           access_token_secret=access_token_secret,
                           wait_on_rate_limit=True)
    return client


def get_timed_tweets_from_api_into_database(client, query, start_date, end_date, collection_name):
    extracted_tweets = dict()


    while start_date < end_date:
        extracted_tweets = client.search_all_tweets(query + " -is:retweet",
                                                    start_time=(end_date - datetime.timedelta(minutes=30)).isoformat(),
                                                    end_time=end_date.isoformat(), max_results=500,
                                                    tweet_fields=["author_id", "conversation_id", "created_at",
                                                                  "possibly_sensitive", "in_reply_to_user_id", "source",
                                                                  "public_metrics", "lang"])
        end_date = end_date - datetime.timedelta(minutes=30)

        filtered_tweets = []
        if extracted_tweets[0]:
            for tweet in extracted_tweets[0]:
                if tweet.lang == "en":
                    filtered_tweets.append(
                        {
                            "id" : tweet.id,
                            "text" : tweet.text,
                            "query" : query,
                            "author_id" : tweet.author_id,
                            "conversation_id" : tweet.conversation_id,
                            "created_at" : tweet.created_at,
                            "possibly_sensitive" : tweet.possibly_sensitive,
                            "in_reply_to_user_id" : tweet.in_reply_to_user_id,
                            "source" : tweet.source,
                            "retweet_count" : tweet.public_metrics.get("retweet_count"),
                            "reply_count" : tweet.public_metrics.get("reply_count"),
                            "like_count" : tweet.public_metrics.get("like_count"),
                            "quote_count" : tweet.public_metrics.get("quote_count")
                        }
                    )
            put_tweets(filtered_tweets, collection_name)


def put_tweets(tweets, collection_name):
    client = pymongo.MongoClient(os.environ.get("MongoDB_CONNECTION_STRING"))
    database = client[os.environ.get("MongoDB_DATABASE_NAME")]
    collection = database[collection_name]
    collection.insert_many(tweets)

