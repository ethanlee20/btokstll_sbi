from helpers.data_prep import prep_data_dir


def main():
    for dir_ in (
        "data/vary_7_9/train",
        "data/vary_7_9/val",
        "data/vary_7_9/val_sets",
    ):
        prep_data_dir(dir_)


if __name__ == "__main__":
    main()
