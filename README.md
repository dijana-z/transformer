# :red_car: Transformer

A neural translator based on the model from [Attention Is All You Need](https://arxiv.org/abs/1706.03762), the paper that introduced a novel approach to sequence processing tasks using Attention Mechanisms instead of the classic RNN based approach. They achieved state-of-the-art results while significantly reducing train time and increasing network throughput.

## :information_source: About:

The code is written in Python and uses [TensorFlow](https://www.tensorflow.org/), a deep learning framework created by Google.
We based our implementation on the model architecture from the paper with a few minor tweaks to accomodate for limited training resources. The architecture is described in the next section.

## :shipit: Model architecture:

Instead of using a standard RNN based approach, the Transformer used Attention Mechanisms to process the input sequences and learn a semantic mapping between the source sentence (in our case English) and the target sentence (German). It can be viewed as a two piece architecture consisting of an **encoder** and **decoder**. Both have the same input processing, an embedding layer followed by positional encoding. Because we aren't using RNNs which naturally encode sequential relations (this makes them slow), we have to manually encode them. This is done via sine-wave transformations that add an intinsic temporal component to the data while allowing it to be processed in whole at any given moment.

* **Encoder** architecture:
  * The **encoder** consists of a variable number of Multi-Head Scaled Dot-Product Attention Blocks (MHDPA), which takes the position encoded data and extracts a key, query, value tuple from it, encodes relations using scaled dot-product attention and outputs a processed sequence.

* **Decoder** architecture:
  * The **decoder** consists of a variable number of a stacked pair of Masked MHDPA and MHDPA blocks. The same type of processing is done here as in the **encoder** the only difference being that the key, value pair from the **encoder** is fed into the MHDPA block and the query is taken from the Masked MHDPA block.
  
Outputs from the decoder pass through a linear layer with a softmax activation to produce the output sequence.

![Model](https://camo.githubusercontent.com/88e8f36ce61dedfd2491885b8df2f68c4d1f92f5/687474703a2f2f696d6775722e636f6d2f316b72463252362e706e67)
## :computer: Running the code:
**TODO**: Write section.

## :mortar_board: Authors:
* Dijana Zulfikaric | dijanaz1996@gmail.com | GitHub &bull; [dijana-z](https://github.com/dijana-z) | LinkedIn &bull; [in/dijana-zulfikaric](https://www.linkedin.com/in/dijana-zulfikaric/)
* Stefan Pantic | stefanpantic13@gmail.com | GitHub &bull; [stefanpantic](https://github.com/stefanpantic) | LinkedIn &bull; [in/stefan-pantic](https://www.linkedin.com/in/stefan-pantic/)
