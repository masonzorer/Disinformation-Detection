# file for creating collections

import pymongo
import os
import pandas as pd
import database
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.server_api import ServerApi

def main():

    # retrieve twitter data from sql database
    cnx = database.connect('Disinformation')
    data = database.get_all_statement(cnx, "SELECT * FROM LabeledTweets;", ())
    cnx.close()

    # put data into dataframe
    df = pd.DataFrame(data, columns=['id', 'text', 'label'])

    # remove id column
    df = df.drop(columns=['id'])

    # remove rows with label 'Unknown'
    df = df[df.label != 'Unknown']

    # change all 'Satire' labels to 'Satire/Joke'
    df['label'] = df['label'].replace(['Satire'], 'Satire/Joke')

    # connect to mongo database
    load_dotenv()
    connection_string = os.getenv("MongoDB_CONNECTION_STRING")
    client = MongoClient(connection_string, server_api=ServerApi('1'))

    # add df to collection called 'Labeled Data'
    mongo_database = client["Disinformation"]
    mongo_collection = mongo_database["Labeled Data"]
    mongo_collection.insert_many(df.to_dict('records'))




if __name__ == "__main__":
    main()