# Train BERT model on the dataset

import transformers
from transformers import get_linear_schedule_with_warmup
import torch
import wandb

#  Define model architecture
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
        
# Function to get accuracy of model
def get_accuracy(model, data, device):
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    correct = 0
    for i, (features, targets) in enumerate(data):
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

    # calculate accuracy and loss
    accuracy = round(correct.item() / total, 3)
    avg_loss = round(total_loss / total, 3)

    return accuracy, avg_loss

def eval_and_log(model, train_data, dev_data, device, epoch):
    # calculate accuracy on training and validation data
    train_accuracy, train_loss = get_accuracy(model, train_data, device)
    dev_accuracy, dev_loss = get_accuracy(model, dev_data, device)

    # Print accuracy
    print(f'Epoch: {epoch+1}, Train Accuracy: {train_accuracy}, Dev Accuracy: {dev_accuracy}')
    print(f'Epoch: {epoch+1}, Train Loss: {train_loss}, Dev Loss: {dev_loss}')

    # log accuracy to W&B
    wandb.log({"Epoch": epoch+1,
        "Train Accuracy": train_accuracy,
        "Dev Accuracy": dev_accuracy,
        "Train Loss": train_loss,
        "Dev Loss": dev_loss})
    
    return dev_loss, dev_accuracy

# Train the model
def train(model_name, device, train_loader, train_data, dev_data, num_epochs, learn_rate, dropout, optimizer='Adam', weight_decay=0, scheduler=False, warmup=0):
    # Define model and loss function
    model = ClassiferModel(dropout)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    # Define optimizer
    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=weight_decay)
    elif optimizer == 'RAdam':
        optimizer = torch.optim.RAdam(model.parameters(), lr=learn_rate, weight_decay=weight_decay)
    elif optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learn_rate, weight_decay=weight_decay)

    # define learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=warmup,
                                                num_training_steps=len(train_loader)*num_epochs)

    # W&B watch on model state
    wandb.watch(model, criterion, log='all')
    lowest_dev_loss = float('inf')
    highest_dev_accuracy = 0.0

    # Initialize metrics
    total_updates = 0
    running_correct = 0
    running_total = 0
    running_loss = 0.0

    # Training Loop
    for epoch in range(num_epochs):
        # Keep track of accuracy
        for i, (features, targets) in enumerate(train_loader):

            # Add batch to GPU
            features = [feature.to(device) for feature in features]
            targets = targets.type(torch.LongTensor)
            targets = targets.to(device)

            # Forward pass and loss calculation
            output = model(features[0], features[1], features[2])
            loss = criterion(output, targets)
            total_updates += 1

            # Keep track of metrics for weights and biases
            # 25 update accuracy
            _, preds = torch.max(output, dim=1)
            running_correct += (preds == targets).float().sum()
            running_total += len(targets)
            # 25 update average loss
            running_loss += loss.item() * len(targets)

            if total_updates % 25 == 0:
                # calculate performance metrics over the last 50 updates
                accuracy = round(running_correct.item() / running_total, 3)
                avg_loss = round(running_loss / running_total, 3)
                print(f"Update: {total_updates}, Avg Train Loss: {avg_loss}, Accuracy: {accuracy}")

                # log metrics to W&B
                wandb.log({"Update": total_updates,        
                           "Avg_Train_Loss": avg_loss,
                           "Accuracy": accuracy,})
                
                # reset metrics
                running_total = 0
                running_correct = 0
                running_loss = 0.0

            # Backward pass and weight optimization
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()

            # Evaluate accuracy after half any epoch
            if i == len(train_loader) // 2:
                with torch.no_grad():
                    print(f"Halfway through Epoch: {epoch+1}, Evaluating Accuracy...")

                    # calculate accuracy on training and validation data
                    dev_loss, dev_accuracy = eval_and_log(model, train_data, dev_data, device, epoch-0.5)

                    # update & save model if dev accuracy is highest
                    if dev_loss < lowest_dev_loss:
                        lowest_dev_loss = dev_loss
                    if dev_accuracy > highest_dev_accuracy:
                        highest_dev_accuracy = dev_accuracy
                        if highest_dev_accuracy > 0.80:
                            torch.save(model.state_dict(), "./models/"+f"M1S2{lowest_dev_loss}-{highest_dev_accuracy}.pt")

        # Evaluate accuracy each epoch
        with torch.no_grad():
            print(f"Epoch: {epoch+1}, Evaluating Accuracy...")

            # calculate accuracy on training and validation data
            dev_loss, dev_accuracy = eval_and_log(model, train_data, dev_data, device, epoch)

            # update & save model if dev accuracy is highest
            if dev_loss < lowest_dev_loss:
                lowest_dev_loss = dev_loss
            if dev_accuracy > highest_dev_accuracy:
                highest_dev_accuracy = dev_accuracy
                if highest_dev_accuracy > 0.80:
                    if (model_name.get()):
                        torch.save(model.state_dict(), "./models/"+model_name.get()+".pt")
                    else:
                        torch.save(model.state_dict(), "./models/"+f"M1S2{lowest_dev_loss}-{highest_dev_accuracy}.pt")

    # log lowest dev loss and highest dev accuracy to W&B
    wandb.log({"Lowest_Dev_Loss": lowest_dev_loss,
               "Highest_Dev_Accuracy": highest_dev_accuracy})

def predict(device, saved_model, data):
    batch_size = 10
    #load the model
    model = ClassiferModel(0)
    model.load_state_dict(torch.load(saved_model, map_location=torch.device(device)))
    model.to(device)
    model.eval()
    
    # get predictions
    with torch.no_grad():
        out = model(data[0].to(device), data[1].to(device), data[2].to(device))
        
    return out
