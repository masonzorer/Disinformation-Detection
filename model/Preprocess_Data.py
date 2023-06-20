# Format loaded data for training and evaluation

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AutoTokenizer

# define the dataset class
class TweetDataset(Dataset):
    def __init__(self, dataframe, max_len):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.data = dataframe
        self.Tweet = dataframe.text
        self.targets = self.data.label
        self.max_len = max_len
    
    def __len__(self):
        return len(self.Tweet)
    
    def __getitem__(self, index):
        Tweet = str(self.Tweet[index])
        Tweet = " ".join(Tweet.split())

        # tokenize the tweet/text
        inputs = self.tokenizer.encode_plus(
            Tweet,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True
        )

        target = int(self.targets[index])
        targets = torch.tensor(target, dtype=torch.float)
        ids = torch.tensor(inputs['input_ids'])
        mask = torch.tensor(inputs['attention_mask'])
        token_type_ids = torch.tensor(inputs["token_type_ids"])

        return [ids, mask, token_type_ids], targets
    
# preprocess data
def preprocess(train_set, dev_set, BATCH_SIZE, MAX_LEN):

    # Create the dataset and dataloader for the neural network
    train_dataset = TweetDataset(train_set, MAX_LEN)
    dev_dataset = TweetDataset(dev_set, MAX_LEN)
    
    # params for mb and train
    train_mb = {'batch_size': BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }
    
    # params for evaluation data
    train_params = {'batch_size': 8,
                'shuffle': True,
                'num_workers': 0
                }

    dev_params = {'batch_size': 8,
                'shuffle': True,
                'num_workers': 0
                }
    
    training_loader = DataLoader(train_dataset, **train_mb)
    train_data = DataLoader(train_dataset, **train_params)
    dev_data = DataLoader(dev_dataset, **dev_params)

    return training_loader, train_data, dev_data

#process a single tweet for a prediction
def preprocess_text(input_text):
    input_text = " ".join(input_text.split())
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer.encode_plus(
        input_text,
        None,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        return_token_type_ids=True
    )
    
    ids = torch.tensor(inputs['input_ids']).unsqueeze(0)
    mask = torch.tensor(inputs['attention_mask']).unsqueeze(0)
    token_type_ids = torch.tensor(inputs["token_type_ids"])
    
    return [ids, mask, token_type_ids]
