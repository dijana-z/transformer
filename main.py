from argparse import ArgumentParser

from transformer import preprocessing


def main():
    # Create argument parser
    argparser = ArgumentParser()
    argparser.add_argument('--train_inputs', type=str, default='./data/train.tags.de-en.en',
                           help='Path to training data inputs.')
    argparser.add_argument('--train_labels', type=str, default='./data/train.tags.de-en.en',
                           help='Path to training data labels.')
    argparser.add_argument('--test_inputs', type=str, default='./data/IWSLT16.TED.tst2014.de-en.en.xml',
                           help='Path to test data inputs.')
    argparser.add_argument('--test_labels', type=str, default='./data/IWSLT16.TED.tst2014.de-en.en.xml',
                           help='Path to training data labels.')
    argparser.add_argument('--en_vocab_path', type=str, default='./data/en-vocab.csv',
                           help='Path to English vocabulary file.')
    argparser.add_argument('--de_vocab_path', type=str, default='./data/de-vocab.csv',
                           help='Path to German vocabulary file.')

    # Parse arguments
    flags = argparser.parse_args()

    # Make vocabularies
    preprocessing.make_vocabulary(flags.train_inputs, flags.en_vocab_path)
    preprocessing.make_vocabulary(flags.train_labels, flags.de_vocab_path)
    train_dataset, test_dataset = preprocessing.create_datasets(flags)


if __name__ == '__main__':
    main()
