
from pandas import read_parquet, concat
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

from helpers.data_prep import prep_data
from helpers.train import run_training_on_datasets
from helpers.model import MLP
from helpers.util import select_device


def main():

    train_dataset = prep_data(
        "data/train_small.parquet"
    )
    eval_dataset = prep_data(
        "data/val_small.parquet"
    )

    device = select_device()

    model = MLP().to(device)

    loss_fn = CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=3e-4)

    losses = run_training_on_datasets(
        train_dataset = train_dataset,
        train_batch_size = 10_000,
        model = model,
        loss_fn = loss_fn,
        optimizer = optimizer,
        num_epochs = 10,
        eval_dataset = eval_dataset,
        eval_batch_size = 10_000,
    )

    breakpoint()

    print()


if __name__ == "__main__":
    main()
    


