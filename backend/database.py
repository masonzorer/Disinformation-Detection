import os
import mysql.connector
from dotenv import load_dotenv
from datetime import date, datetime, timedelta
from pathlib import Path
import pandas as pd

# Connect to a database given the name dbname, and return the connection object
def connect(dbname):
    # Get the port and password from the dotenv file
    file_exists = os.path.exists('.env')
    if file_exists:
        load_dotenv()
    else:
        load_dotenv(dotenv_path = Path('../.env'))
    

    dbhost = os.getenv("SQL_HOST")
    dbpass = os.getenv("SQL_PASS")
    dbuser = os.getenv("SQL_USER")

    # connect to the database
    try:
        cnx = mysql.connector.connect(user=dbuser, 
                                    password=dbpass,
                                    host=dbhost,
                                    database=dbname)
    except:
        print("Failed to connect to database")
        exit(1)

    # return the connection to the database
    return cnx

# Query the database and return the result as a pandas dataframe
def query_to_dataframe(cnx, query):
    #query = "Select * from LabeledTweets;"
    try:
        result_dataFrame = pd.read_sql(query,cnx)
    except:
        print("Query failed")
        exit(1)
    return result_dataFrame

# Insert a tweet into the database
def insert_statement(cnx, statement, values):
    cursor = cnx.cursor()
    cursor.execute(statement, values)
    cnx.commit()
    cursor.close()
    
# Pull a single tweet from the database
def get_statement(cnx, statement, values):
    cursor = cnx.cursor(buffered=True)
    cursor.execute(statement, values)
    ret = cursor.fetchone()
    cursor.close()
    return ret

# Pull all tweets from the database
def get_all_statement(cnx, statement, values):
    cursor = cnx.cursor(buffered=True)
    cursor.execute(statement, values)
    ret = cursor.fetchall()
    cursor.close()
    return ret

# Get a single tweet from the database that has not been labeled
def get_unlabeled_tweet(cnx):
    cursor = cnx.cursor()
    query = ('SELECT id, text FROM Tweets WHERE id NOT IN (SELECT id FROM LabeledTweets) ORDER BY RAND() LIMIT 1')
    cursor.execute(query)
    current_tweet = cursor.fetchone()
    cursor.close()
    return current_tweet

# Add a tweet's data to the database
def insertTweet(cnx, tweet):
    # create a cursor, this object can execute operations such as sql statements
    cursor = cnx.cursor()

    # This is an example statement that will add a tweet to the database:
    add_tweet_statement = ("INSERT INTO Tweets "
                           "(id, text, query, author_id, conversation_id, created_at, possibly_sensitive, in_reply_to_user_id, source, retweet_count, reply_count, like_count, quote_count) "
                           "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)")

    # Put the tweet data into a tuple
    tweet_data = (
    tweet[0], tweet[1], tweet[2], tweet[3], tweet[4], tweet[5], tweet[6], tweet[7], tweet[8], tweet[9], tweet[10],
    tweet[11], tweet[12])

    # Insert the example tweet
    cursor.execute(add_tweet_statement, tweet_data)
    cnx.commit()

    # close the cursor
    cursor.close()


# Get a tweet from the database
def getTweet(cnx):
    # create a cursor, this object can execute operations such as sql statements
    cursor = cnx.cursor()

    # An example statement to show how we might get data, this gets all tweets posted between two dates:
    query = ("SELECT text, retweet_count, like_count FROM Tweets WHERE date_posted BETWEEN %s AND %S")

    date_start = datetime.date(2020, 1, 1)
    date_end = datetime.date(2020, 12, 31)

    cursor.execute(query, (date_start, date_end))

    # print the data we retrieved:
    for (text, retweets, likes) in cursor:
        print("{} has {} retweets and {} likes".format(text, retweets, likes))

    # close the cursor
    cursor.close()


# closes the connection to a database
def disconnect(cnx):
    cnx.close()
