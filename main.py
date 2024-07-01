import os

import constants
from data.StartingDataset import StartingDataset
from networks.StartingNetwork import StartingNetwork
from train_functions.starting_train import starting_train
import numpy as np


def main():
    # Get command line arguments
    hyperparameters = {"epochs": constants.EPOCHS, "batch_size": constants.BATCH_SIZE}

    # TODO: Add GPU support. This line of code might be helpful.
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Epochs:", constants.EPOCHS)
    print("Batch size:", constants.BATCH_SIZE)

    # Initalize dataset and model. Then train the model!
    train_dataset = StartingDataset(constants.TRAIN_CSV_PATH, constants.TRAIN_IMG_PATH)
    val_dataset = StartingDataset(constants.TEST_CSV_PATH, constants.TEST_IMG_PATH, training_set=False)
    model = StartingNetwork()
    # ex = train_dataset[1]
    # ex = np.array(ex[0])
    # print(len(ex))
    # print(len(ex[0]))
    # print(len(ex[0][0]))
    # for i in range(len(ex)):
    #     for j in range(len(ex[0])):
    #         print(ex[i][j])
    starting_train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        hyperparameters=hyperparameters,
        n_eval=constants.N_EVAL,
    )


if __name__ == "__main__":
    main()