from argparse import ArgumentParser

from transformer import preprocessing
from transformer.model import Transformer


def main():
    # Create argument parser
    argparser = ArgumentParser()

    # File flags
    argparser.add_argument('--train_inputs', type=str, default='./data/train.tags.de-en.en',
                           help='Path to training data inputs.')
    argparser.add_argument('--train_labels', type=str, default='./data/train.tags.de-en.en',
                           help='Path to training data labels.')
    argparser.add_argument('--val_inputs', type=str, default='./data/IWSLT16.TED.tst2012.de-en.en.xml',
                           help='Path to val data inputs.')
    argparser.add_argument('--val_labels', type=str, default='./data/IWSLT16.TED.tst2012.de-en.en.xml',
                           help='Path to val data labels.')
    argparser.add_argument('--test_inputs', type=str, default='./data/IWSLT16.TED.tst2014.de-en.en.xml',
                           help='Path to test data inputs.')
    argparser.add_argument('--test_labels', type=str, default='./data/IWSLT16.TED.tst2014.de-en.en.xml',
                           help='Path to test data labels.')
    argparser.add_argument('--en_vocab_path', type=str, default='./data/en-vocab.csv',
                           help='Path to English vocabulary file.')
    argparser.add_argument('--de_vocab_path', type=str, default='./data/de-vocab.csv',
                           help='Path to German vocabulary file.')

    # Model hyperparameters
    argparser.add_argument('--logdir', type=str, default='./tmp/model',
                           help='Path to model checkpoint directory.')
    argparser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train for.')
    argparser.add_argument('--batch_size', type=int, default=16, help='Size of each batch.')
    argparser.add_argument('--learning_rate', type=float, default=1e-4, help='Parameter update rate.')
    argparser.add_argument('--encoder_blocks', type=int, choices=range(1, 7),
                           help='Number of MHDPA blocks to use in encoder.')
    argparser.add_argument('--decoder_blocks', type=int, choices=range(1, 7),
                           help='Number of Masked MHDPA + MHDPA blocks to use in decoder')

    # Parse arguments
    flags = argparser.parse_args()

    # Make vocabularies
    preprocessing.make_vocabulary(flags.train_inputs, flags.en_vocab_path)
    preprocessing.make_vocabulary(flags.train_labels, flags.de_vocab_path)
    (x_train, y_train), (x_val, y_val), _ = preprocessing.create_datasets(flags)

    # Create model
    model = Transformer(flags)
    model.fit(x_train, y_train, x_val, y_val)


if __name__ == '__main__':
    main()
