from tkinter import *
#import webview
from functools import partial
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from pymongo.server_api import ServerApi

# globals
session_count = 0
load_dotenv()
connection_string = os.getenv("MongoDB_CONNECTION_STRING")
client = MongoClient(connection_string, server_api=ServerApi('1'))
collection = client['Disinformation']["Labeled Data"]

def setup():
    
    # GUI setup
    root = Tk()
    root.title('Classification Tool')
    root.geometry('850x400')
    lbl = Label(root, text = 'Create a post:', font=('Helvetica', 20))
    lbl.grid(sticky = W, pady=10, padx=10)
    lbl2 = Label(root, text = 'No Selection')
    lbl2.grid(row=4, columnspan=5, pady=40)

    #create a text input box
    text_box = Text(root, height=10, width=100)
    text_box.grid(row=2, columnspan=5, pady=15, padx=10)
    
    #make counters for session and total:
    totalLabeled = collection.count_documents({})
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



    # Submit button
    #get text from text box
    submit = partial(submit_selection, text_box, lbl2)
    Button(root, 
        text = 'Submit' ,
        fg = 'black', 
        command=submit
        ).grid(row=5, columnspan=5)
    
    root.mainloop()

def button_press(button, lbl2):
    lbl2.configure(text = button)

# Submit button action
def submit_selection(text, lbl2):
    if lbl2.cget("text") == 'No Selection':
        return
    document = {"text": text.get("1.0",END).strip(), "label": lbl2.cget("text")}
    print(document)
    collection.insert_one(document)
    text.delete('1.0', END)
    lbl2.configure(text = 'No Selection')
    


if __name__=='__main__':
    setup()

