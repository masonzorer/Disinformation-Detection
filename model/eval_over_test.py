# get the accuracy of the model on the test data
import Preprocess_Data
import transformers
import pandas as pd
from sklearn.metrics import confusion_matrix
import sklearn.metrics
import torch
import matplotlib.pyplot as plt
import seaborn as sn


def main():
    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # create class for the model
    class ClassiferModel(torch.nn.Module):
        def __init__(self, dropout):
            super(ClassiferModel, self).__init__()
            self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
            self.dropout = torch.nn.Dropout(dropout)
            self.l2 = torch.nn.Linear(768, 4)
    
        def forward(self, ids, mask, token_type_ids):
            _, output = self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
            output = self.dropout(output)
            output = self.l2(output)
            return output

    # load the model
    model = ClassiferModel(0.4)
    model.load_state_dict(torch.load("./models/"+"-M1S20.448-0.868.pt"))
    model.to(device)
    model.eval()

    # load the test data
    BATCH_SIZE = 2
    MAX_LEN = 512
    test_dataset = pd.read_csv("./data/test2.csv")
    test_loader, t, t1 = Preprocess_Data.preprocess(test_dataset, test_dataset, BATCH_SIZE, MAX_LEN)

    # get the accuracy of the model on the test data
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    correct = 0

    # keep track of predictions and targets
    predictions = []
    targets1 = []

    for i, (features, targets) in enumerate(test_loader):
        features = [feature.to(device) for feature in features]
        targets = targets.type(torch.LongTensor)
        targets = targets.to(device)
        outputs = model(features[0], features[1], features[2])
        # loss calculation
        loss = criterion(outputs, targets)
        total_loss += loss.item() * len(targets)
        # accuracy calculation
        _, preds = torch.max(outputs, dim=1)
        correct += (preds == targets).float().sum()
        total += len(targets)

        # add predictions and targets to lists
        predictions.extend(preds.tolist())
        targets1.extend(targets.tolist())

    # calculate accuracy and loss
    accuracy = round(correct.item() / total, 3)
    avg_loss = round(total_loss / total, 3)

    print(f'Test Accuracy: {accuracy}')
    print(f'Test Loss: {avg_loss}')

    # create confusion matrix
    confusion_matrix = sklearn.metrics.confusion_matrix(targets1, predictions)
    print(confusion_matrix)

    # graph confusion matrix
    df_cm = pd.DataFrame(confusion_matrix, index = [i for i in ["Disinformation", "Misinformation", "Satire/Joke", "None"]],
                    columns = [i for i in ["Disinformation", "Misinformation", "Satire/Joke", "None"]])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, cmap="Blues", fmt='g')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()




if __name__ == "__main__":
    main()