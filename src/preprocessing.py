import codecs
import csv
import os
from collections import Counter

import regex


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
    with codecs.open(infile, 'r', 'utf-8') as f:
        text = regex.sub('[^\s\p{Latin}\']', '', f.read())

    # Count words in corpora
    word_counts = Counter(text.split())
    if not os.path.exists('preprocessed'):
        os.mkdir('preprocessed')

    # Write sorted word index to outfile
    with codecs.open(f'./preprocessed/{outfile}', 'w', 'utf-8') as f:
        writer = csv.DictWriter(f, ['TOKEN', 'FREQUENCY'])
        writer.writeheader()
        for word, count in word_counts.most_common(len(word_counts)):
            writer.writerow({'TOKEN': word, 'FREQUENCY': count})

#
# if __name__ == '__main__':
#     make_vocabulary('../corpora/train.tags.de-en.de', 'out.csv')
