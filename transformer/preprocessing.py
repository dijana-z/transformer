import codecs
import csv
import os
from collections import Counter

import regex
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences


def make_vocabulary(infile, outfile):
    """Creates vocabulary.

    Parameters
    ----------
    infile:
        Input file name.
    outfile:
        Output file name.
    """
    # Read raw data from infile
    if os.path.exists(outfile):
        return

    with codecs.open(infile, 'r', 'utf-8') as f:
        text = regex.sub('[^\s\p{Latin}\']', '', f.read())

    # Count words in corpora
    word_counts = Counter(text.split())

    # Create nested directory
    if not os.path.exists(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))

    # Write sorted word index to outfile
    with codecs.open(outfile, 'w', 'utf-8') as f:
        writer = csv.DictWriter(f, ['TOKEN', 'FREQUENCY'])
        writer.writeheader()
        writer.writerows(
            [{'TOKEN': token, 'FREQUENCY': 10000000} for token in ['<PAD>', '<UNK>', '<S>', '</S>']])
        for word, count in word_counts.most_common(len(word_counts)):
            writer.writerow({'TOKEN': word, 'FREQUENCY': count})


def load_vocabulary(infile, min_occurrences=50):
    """Loads vocabulary.

    Parameters
    ----------
    infile:
        Input file.
    min_occurrences:
        Minimum number of times a word has to occur to be added to vocabulary

    Returns
    -------
        indexed_words, worded_indexes: Indexed words and vice versa.
    """
    with codecs.open(infile, 'r', 'utf-8') as f:
        reader = csv.DictReader(f)
        words = [row['TOKEN'] for row in reader if int(row['FREQUENCY']) >= min_occurrences]

    indexed_words = {word: index for index, word in enumerate(words)}
    worded_indexes = {index: word for word, index in indexed_words.items()}

    return indexed_words, worded_indexes


def create_data(source_sentences, target_sentences, en_vocab_file, de_vocab_file, maxlen=15):
    """Creates encoded sentences.

    Parameters
    ----------
    source_sentences:
        Source sentences.
    target_sentences:
        Target sentences.
    en_vocab_file:
        File with english vocabulary.
    de_vocab_file:
        File with german vocabulary.
    maxlen:
        Size of all output sequences

    Returns
    -------
        Labeled data.
    """
    en_wti, en_itw = load_vocabulary(en_vocab_file)
    de_wti, de_itw = load_vocabulary(de_vocab_file)
    x_data, y_data, sources, targets = [], [], [], []

    for source, target in zip(source_sentences, target_sentences):
        x = [en_wti.get(word, 1) for word in source.split()] + [en_wti['</S>']]
        y = [de_wti.get(word, 1) for word in target.split()] + [de_wti['</S>']]

        if max(len(x), len(y)) < maxlen:
            x_data.append(x)
            y_data.append(y)
            sources.append(source)
            targets.append(target)

    x_data = pad_sequences(x_data, maxlen=maxlen, padding='post')
    y_data = pad_sequences(y_data, maxlen=maxlen, padding='post')

    return x_data, y_data, sources, targets


def load_data(input_file, label_file, en_vocab_file, de_vocab_file, sentence_fn):
    """Loads and parses dataset.
    
    Parameters
    ----------
    input_file:
        Path to dataset inputs.
    label_file:
        Path to dataset labels.
    en_vocab_file:
        Path to file with english vocabulary.
    de_vocab_file:
        Path to file with german vocabulary,
    sentence_fn:
        Callback that parses input files.

    Returns
    -------
        Encoded input and label sentences.
    """
    en_sentences = sentence_fn(input_file)
    de_sentences = sentence_fn(label_file)
    return create_data(en_sentences, de_sentences, en_vocab_file, de_vocab_file)


def create_train_sentences(infile):
    """Creates sentences from data in passed file."""
    return [regex.sub('[^\s\p{Latin}\']', '', line) for line in
            codecs.open(infile, 'r', 'utf-8').read().split('\n') if
            line and not line.startswith('<')]


def create_test_sentences(infile):
    """Create sentences from data in passed file."""

    def refine(line):
        line = regex.sub('<[^>]+>', '', line)
        line = regex.sub('[^\s\p{Latin}\']', '', line)
        return line.strip()

    return [refine(line) for line in codecs.open(infile, 'r', 'utf-8').read().split('\n') if
            line and not line.startswith('<seg')]


def create_datasets(flags):
    """Create tf.data iterators for dataset.

    flags:
        Namespace object with parameters.

    Returns
    -------
        train_dataset, test_dataset: Iterators for train and test datasets.
    """
    # Load data
    x_train, y_train, *_ = load_data(flags.train_inputs, flags.train_labels, flags.en_vocab_path,
                                     flags.de_vocab_path, create_train_sentences)
    x_test, y_test, *_ = load_data(flags.test_inputs, flags.test_labels, flags.en_vocab_path,
                                   flags.de_vocab_path, create_test_sentences)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    return train_dataset, test_dataset
