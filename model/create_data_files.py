# methods for creating data files for the model
import Get_Data

def create_files():
    # load data
    all_data = Get_Data.setup()

    # shuffle data
    all_data = all_data.sample(frac=1).reset_index(drop=True)

    # split data into training, validation and test sets
    train_size = 0.7
    train_dataset = all_data.sample(frac=train_size,random_state=200)
    dev_dataset = all_data.drop(train_dataset.index).reset_index(drop=True)
    test_dataset = dev_dataset.sample(frac=0.5, random_state=200)
    dev_dataset = dev_dataset.drop(test_dataset.index).reset_index(drop=True)

    # reset index
    train_dataset = train_dataset.reset_index(drop=True)
    dev_dataset = dev_dataset.reset_index(drop=True)
    test_dataset = test_dataset.reset_index(drop=True)

    print(f"FULL Dataset: {all_data.shape}")
    print(f"TRAIN Dataset: {train_dataset.shape}")
    print(f"DEV Dataset: {dev_dataset.shape}")
    print(f"TEST Dataset: {test_dataset.shape}")

    # save data
    train_dataset.to_csv("./data/train2.csv", index=False)
    dev_dataset.to_csv("./data/dev2.csv", index=False)
    test_dataset.to_csv("./data/test2.csv", index=False)

def main():
    create_files()

if __name__ == "__main__":
    main()