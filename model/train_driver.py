# main driver for training the model
import Track
import Preprocess_Data
import NLP_Main
import wandb
import torch
from torch import cuda
import pandas as pd

def main(max_len, batch_size, epochs, learn_rate, dropout, optimizer, scheduler, warmup_steps, weight_decay, train_file, dev_file, model_name, num_sweep, sweep = False):
    
    device = 'cuda' if cuda.is_available() else 'cpu'
    print(f"GPU available: {cuda.is_available()}")
    print(f"device: {device}")

    # Hyperparameters
    if (not sweep and max_len.get()):
        MAX_LEN = max_len.get()
    else:
        MAX_LEN = 512
    if (not sweep and batch_size.get()):
        BATCH_SIZE = int(batch_size.get())
    else:
        BATCH_SIZE = 2
    if (not sweep and epochs.get()):
        EPOCHS = int(epochs.get())
    else:
        EPOCHS = 3
    if (not sweep and learn_rate.get()):
        LEARNING_RATE = int(learn_rate.get())
    else:
        LEARNING_RATE = 1e-5
    if (not sweep and dropout.get()):
        DROPOUT = float(dropout.get())
    else:
        DROPOUT = 0.5
    if (not sweep and optimizer.get()):
        OPTIMIZER = float(optimizer.get())
    else:
        OPTIMIZER = 'AdamW'
    if (not sweep and scheduler.get()):
        SCHEDULER = scheduler.get()
    else:
        SCHEDULER = False
    if (not sweep and warmup_steps.get()):
        WARMUP_STEPS = int(warmup_steps.get())
    else:
        WARMUP_STEPS = 0
    if (not sweep and weight_decay.get()):
        WEIGHT_DECAY = float(weight_decay.get())
    else:
        WEIGHT_DECAY = 0.0001

    # Will this be a sweep?
    SWEEP = sweep
    ADDITIONAL_COMPUTER = False
    if (sweep and num_sweep.get()):
        NUM_SWEEPS = int(num_sweep.get())
    else:
        NUM_SWEEPS = 100
    
    # read data from csv files in data directory
    if (train_file.get()):
        train_dataset = pd.read_csv("src/model/data/" + train_file.get() + ".csv")
    else:
        train_dataset = pd.read_csv("src/model/data/train2.csv")
    if (dev_file.get()):
        dev_dataset = pd.read_csv("src/model/data/" + dev_file.get() + ".csv")
    else:
        dev_dataset = pd.read_csv("src/model/data/dev2.csv")


    # Preprocess data into format for BERT
    train_loader, train_data, dev_data = Preprocess_Data.preprocess(train_dataset, dev_dataset, BATCH_SIZE, MAX_LEN)

    # Initialize weights and biases tracking
    if ADDITIONAL_COMPUTER == False:
        sweep_id = Track.initialize_tracking(SWEEP, EPOCHS, LEARNING_RATE, BATCH_SIZE, DROPOUT, OPTIMIZER, SCHEDULER, WARMUP_STEPS)
    
    # define sweep function
    def sweep():
        wandb.init()
        NLP_Main.train(device, train_loader, train_data, dev_data, 
                        wandb.config.epochs, wandb.config.learn_rate, wandb.config.dropout,
                        wandb.config.optimizer, wandb.config.weight_decay)

    # if additional computer, run sweep
    if ADDITIONAL_COMPUTER:
        sweep()
        return

    # train the model
    if SWEEP:
        wandb.agent(sweep_id, function=sweep, count=NUM_SWEEPS)
    else:
        NLP_Main.train(model_name, device, train_loader, train_data, dev_data, EPOCHS, LEARNING_RATE, DROPOUT, OPTIMIZER, WEIGHT_DECAY)



if __name__ == "__main__":
    main()