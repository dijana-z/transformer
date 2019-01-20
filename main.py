import numpy as np
from argparse import ArgumentParser

from transformer import preprocessing
from transformer.model import Transformer


def main():
    # Create argument parser
    argparser = ArgumentParser()

    # File flags
    argparser.add_argument('--train_inputs', type=str, default='./data/train.tags.de-en.en',
                           help='Path to training data inputs.')
    argparser.add_argument('--train_labels', type=str, default='./data/train.tags.de-en.de',
                           help='Path to training data labels.')
    argparser.add_argument('--val_inputs', type=str, default='./data/IWSLT16.TED.tst2012.de-en.en.xml',
                           help='Path to val data inputs.')
    argparser.add_argument('--val_labels', type=str, default='./data/IWSLT16.TED.tst2012.de-en.de.xml',
                           help='Path to val data labels.')
    argparser.add_argument('--test_inputs', type=str, default='./data/IWSLT16.TED.tst2014.de-en.en.xml',
                           help='Path to test data inputs.')
    argparser.add_argument('--test_labels', type=str, default='./data/IWSLT16.TED.tst2014.de-en.de.xml',
                           help='Path to test data labels.')
    argparser.add_argument('--en_vocab_path', type=str, default='./data/en-vocab.csv',
                           help='Path to English vocabulary file.')
    argparser.add_argument('--de_vocab_path', type=str, default='./data/de-vocab.csv',
                           help='Path to German vocabulary file.')

    # Model hyperparameters
    argparser.add_argument('--logdir', type=str, default='./tmp/model',
                           help='Path to model checkpoint directory.')
    argparser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train for.')
    argparser.add_argument('--batch_size', type=int, default=32, help='Size of each batch.')
    argparser.add_argument('--learning_rate', type=float, default=1e-4, help='Parameter update rate.')
    argparser.add_argument('--mhdpa_heads', type=int, default=8, help='Number of heads in MHDPA module.')
    argparser.add_argument('--mlp_units', type=int, default=512, help='Number of MLP units.')
    argparser.add_argument('--mhdpa_blocks', type=int, default=6, choices=range(1, 10),
                           help='Number of MHDPA blocks to use in encoder.')
    argparser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate.')
    argparser.add_argument('--sequence_length', type=int, default=15,
                           help='Length of input/output sequences.')

    # Running parameters
    argparser.add_argument('--mode', type=str, choices=['train', 'test', 'predict'], default='train',
                           help='Which mode to run script in.')

    # Parse arguments
    flags = argparser.parse_args()

    # Make vocabularies
    en_vocab_size = preprocessing.make_vocabulary(flags.train_inputs, flags.en_vocab_path)
    de_vocab_size = preprocessing.make_vocabulary(flags.train_labels, flags.de_vocab_path)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = preprocessing.create_datasets(flags)

    # Create model
    model = Transformer(flags, en_vocab_size, de_vocab_size)

    if flags.mode == 'train':
        model.fit(x_train, y_train, x_val, y_val)
    elif flags.mode == 'test':
        loss, acc = model.eval(x_test, y_test)
        print(f'[loss: {loss}; acc: {acc}]')
    elif flags.mode == 'predict':
        indices = np.random.randint(0, len(x_test), size=flags.batch_size)
        model.predict(x_test[indices], y_test[indices])

if __name__ == '__main__':
    main()
