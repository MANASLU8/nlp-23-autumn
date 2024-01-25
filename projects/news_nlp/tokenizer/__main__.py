from tokenizer import process_file


def main():
    fname_train = '.../assets/raw-dataset/train.csv'
    fname_test = '.../assets/raw-dataset/test.csv'
    process_file(fname_train)
    process_file(fname_test)


if __name__ == "__main__":
    main()
