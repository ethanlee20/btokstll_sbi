from helpers.data_prep import prep_data_dir


def main():
    for dir_ in (
        #    "data/train_sm",
        #    "data/train_vary",
        "data/val_sm",
        "data/val_vary",
        #    "data/val"
    ):
        prep_data_dir(dir_)


if __name__ == "__main__":
    main()
