import os
from tkinter import *
import webview
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
from functools import partial

'''
There is commented out code in this file that uses our initial database
'''

# globals
current_tweet = None
entry_values = None
session_count = 0

def setup():
    
    # GUI setup
    root = Tk()
    root.title('Classification Tool')
    root.geometry('850x400')
    lbl = Label(root, text = 'Classify this post:', font=('Helvetica', 20))
    lbl.grid(sticky = W, pady=10, padx=10)
    tweet = Label(root, text = 'Tweet', wraplength=800, justify="center")
    tweet.grid(row=2, columnspan=5, pady=15, padx=10)
    lbl2 = Label(root, text = 'No Selection')
    lbl2.grid(row=4, columnspan=5, pady=40)
    
    # make counters for session and total:

    # connect to mongo database
    load_dotenv()
    connection_string = os.getenv("MongoDB_CONNECTION_STRING")
    client = MongoClient(connection_string, server_api=ServerApi('1'))
    database = client["Disinformation"]

    # define collections
    unlabeled_collection = database["Unlabeled Data"]
    labeled_collection = database["Labeled Data"]

    totalLabeled = labeled_collection.count_documents({})

    # create labels for counters
    tLbl = Label(root, text = 'Total Labeled Tweets: ' + str(totalLabeled), font=('Helvetica', 10))
    cLbl = Label(root, text = 'Session Labeled Tweets: ' + str(session_count), font=('Helvetica', 10))
    tLbl.grid(row=6, column = 0, columnspan=1, pady=40, padx=10)
    cLbl.grid(row=6, column = 1, columnspan=5, pady=40, padx=10)

    # setup buttons
    buttons = ['Disinformation', 'Misinformation', 'Satire/Joke', 'None', 'Unknown']
    for i in range(5):
        text = buttons[i]
        press = partial(button_press, text, lbl2)
        button = Button(root, 
                text = text, 
                fg = 'black', 
                command=press)
        button.grid(column=i, row=3)

    
    # reopen twitter page button
    Button(root, 
        text = 'Reload Webpage' ,
        fg = 'black', 
        command=display_tweet
        ).grid(column=0, row=1, sticky = W, padx=10)


    # Submit button
    submit = partial(submit_selection, labeled_collection, unlabeled_collection, lbl2, tLbl, cLbl, tweet)
    Button(root, 
        text = 'Submit' ,
        fg = 'black', 
        command=submit
        ).grid(row=5, columnspan=5)

    # get first tweet
    get_new_tweet(tweet, unlabeled_collection)

    root.mainloop()
    

def get_new_tweet(tweet, collection):
    global current_tweet

    # get new tweet
    current_tweet = collection.find_one_and_delete({})

    display_tweet()
    tweet.configure(text = current_tweet['text'])

# Display Tweet
def display_tweet():
    webview.create_window('Online Post', 'https://twitter.com/ur/status/' + str(current_tweet['id']))
    webview.start()

def button_press(button, lbl2):
    global entry_values
    entry_values = (current_tweet['text'], button)
    lbl2.configure(text = button)

# Submit button action
def submit_selection(labeled_collection, unlabeled_collection, lbl2, tLbl, cLbl, tweet):
    global entry_values, session_count
    if entry_values:
        # add tweet to labeled collection
        print(entry_values)
        labeled_collection.insert_one({'text': entry_values[0], 'label': entry_values[1]})

        # reset entry values
        entry_values = None
        session_count += 1
        totalLabeled = labeled_collection.count_documents({})

        tLbl.configure(text = 'Total Labeled Tweets: ' + str(totalLabeled))
        cLbl.configure(text = 'Session Labeled Tweets: ' + str(session_count))

        # refresh UI & get new tweet
        get_new_tweet(tweet, unlabeled_collection)
        lbl2.configure(text = 'No Selection')


if __name__=='__main__':
    setup()
