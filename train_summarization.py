from argparse import ArgumentParser
from collections import Counter

import numpy as np
import ray
import regex
from modin import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.contrib.framework import nest

from transformer.model import Transformer, predict
from transformer.preprocessing import word_count_to_vocabulary, create_summary_data


def main():
    # Create argument parser
    argparser = ArgumentParser()

    # File flags
    argparser.add_argument('--input_data', type=str, default='./data/amazon/reviews.csv',
                           help='Path to input dataset.')
    argparser.add_argument('--vocabulary_file', type=str, default='./data/summary_vocab.csv')
    argparser.add_argument('--validation_split', type=str, default=0.1, help='Portion of data to use for validation.')

    # Model hyperparameters
    argparser.add_argument('--logdir', type=str, default='./tmp/model',
                           help='Path to model checkpoint directory.')
    argparser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train for.')
    argparser.add_argument('--batch_size', type=int, default=32, help='Size of each batch.')
    argparser.add_argument('--learning_rate', type=float, default=1e-4, help='Parameter update rate.')
    argparser.add_argument('--mhdpa_heads', type=int, default=8, help='Number of heads in MHDPA module.')
    argparser.add_argument('--mlp_units', type=int, default=512, help='Number of MLP units.')
    argparser.add_argument('--mhdpa_blocks', type=int, default=6, choices=range(1, 10),
                           help='Number of MHDPA blocks to use in encoder.')
    argparser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate.')
    argparser.add_argument('--input_sequence_length', type=int, default=15,
                           help='Length of input sequences.')
    argparser.add_argument('--output_sequence_length', type=int, default=15,
                           help='Length of output sequences.')
    # Running parameters
    argparser.add_argument('--mode', type=str, choices=['train', 'test', 'predict'], default='train',
                           help='Which mode to run script in.')

    # Parse arguments
    flags = argparser.parse_args()

    # Load raw data
    data_frame = pd.read_csv(flags.input_data)
    raw_data = data_frame[['Text', 'Summary']].values[:20000, :]
    raw_inputs, raw_labels = np.transpose(raw_data, [1, 0])

    @ray.remote
    def refine(x):
        return regex.sub('[^\s\p{Latin}\']', '', str(x))

    def sentences_to_words(x):
        return nest.flatten(list(map(lambda y: y.split(), x)))

    # Prepare data for network
    raw_inputs = ray.get(list(map(refine.remote, raw_inputs)))
    raw_labels = ray.get(list(map(refine.remote, raw_labels)))
    words = sentences_to_words(raw_inputs) + sentences_to_words(raw_labels)
    word_counts = Counter(words)
    vocab_size = word_count_to_vocabulary(word_counts, flags.vocabulary_file)
    inputs, labels, reviews, summaires = create_summary_data(raw_inputs, raw_labels,
                                                             vocab_file=flags.vocabulary_file,
                                                             input_maxlen=flags.input_sequence_length,
                                                             output_maxlen=flags.output_sequence_length)
    train_inputs, val_inputs, train_labels, val_labels = train_test_split(inputs, labels,
                                                                          test_size=flags.validation_split,
                                                                          random_state=42)

    # Create model
    model = Transformer(flags, vocab_size, vocab_size)

    if flags.mode == 'train':
        model.fit(train_inputs, train_labels, val_inputs, val_labels)
    elif flags.mode == 'test':
        loss, acc = model.eval(val_inputs, val_labels)
        print(f'[loss: {loss}; acc: {acc}]')
    elif flags.mode == 'predict':
        indices = np.random.randint(0, len(val_inputs), size=flags.batch_size)
        inputs, true_outputs, predicted_outputs = predict(model, logdir=flags.logdir, inputs=val_inputs[indices],
                                                          labels=val_labels[indices], vocab_file=flags.vocabulary_file,
                                                          input_seq_len=flags.input_sequence_length,
                                                          output_seq_len=flags.output_sequence_length)
        # noinspection PyTypeChecker
        for s, o, p in zip(inputs, true_outputs, predicted_outputs):
            print('input:     ', s)
            print('true:      ', o)
            print('predicted: ', p)


if __name__ == '__main__':
    main()
