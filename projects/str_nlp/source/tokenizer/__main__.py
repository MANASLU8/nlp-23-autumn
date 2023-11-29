from tokenizer import process_file


def main():
    fname_train = 'C:/Users/n.strokova/Pictures/ITMO/nlp/assets/raw-dataset/train.csv'
    fname_test = 'C:/Users/n.strokova/Pictures/ITMO/nlp/assets/raw-dataset/test.csv'
    process_file(fname_train)
    process_file(fname_test)


if __name__ == "__main__":
    main()
