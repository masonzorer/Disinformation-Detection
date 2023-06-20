# configuration for weights and biases logging

import wandb

def initialize_tracking(sweep, epochs, learn_rate, batch_size, dropout, optimizer, scheduler, warmup):

    # single run configuration
    config={
        "epochs": epochs,
        "learn_rate": learn_rate,
        "batch_size": batch_size,
        "dropout": dropout,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "warmup": warmup
    }

    # sweep configuration
    sweep_config = {
        "program": "train_driver.py",
        "method": "random",
        "metric": {
            "name": "Lowest_Dev_Loss",
            "goal": "minimize",
        },  
        "parameters": {
            "epochs": {
                "value": 3
            },
            "learn_rate": {
                "values": [1e-5, 2e-5, 3e-5]
            },
            "batch_size": {
                "value": 2
            },
            "dropout": {
                "values": [0.2, 0.3, 0.4, 0.5]
            },
            "optimizer": {
                "values": ["Adam", "AdamW"]
            },
            "weight_decay": {
                "values": [0, 1e-3, 1e-4, 1e-5]
            }
        }
    }

    # initialize weights and biases
    if sweep:
        sweep_id = wandb.sweep(sweep=sweep_config, entity="disinformation", project="Disinformation-Classifier")
    else:
        wandb.init(entity="disinformation", project="Disinformation-Classifier", config=config)
        sweep_id = None

    return sweep_id
