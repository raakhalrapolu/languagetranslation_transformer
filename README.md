# Language Translation on WMT 22 Dataset

## Abstract
This project presents an in-depth analysis of advanced neural machine translation techniques using the WMT 2022 dataset, with a focus on English-Czech, English-German, and English-Ukrainian language pairs. We explore two leading machine translation models, Marian MT and T5, along with a custom-built sequence-to-sequence model with an attention mechanism. Our study emphasizes the role of attention-based word embeddings in enhancing translation accuracy, with the T5 model demonstrating superior performance.

## Dataset
The WMT 2022 dataset is a comprehensive collection featuring various domains such as conversations, news, and social media, across multiple language pairs. We specifically utilized three parallel corpora for our translation tasks:
- Czech to/from English: 364,578 sentences
- Ukrainian to/from English: 240,000 sentences
- German to/from English: 422,392 sentences

## Data Pre-processing
Key steps in our data preprocessing include:
- Data cleaning and tokenization.
- Utilizing SentencePiece and Byte Pair Encoding (BPE) techniques.
- Implementation of the T5 tokenizer.

## Software Architecture
Our system integrates a React front-end with a Flask API backend. The architecture ensures seamless translation processes and interactions.

## Model Description
### Marian MT
- Advanced neural network architecture for capturing linguistic nuances.
- Extensive language support and customization options.
- Efficient processing capabilities for large-scale translations.

### T5 Model
- Structured approach for training, including data preparation, model training, validation, and saving.
- The 't5-small' variant is employed for a balance of efficiency and performance.

### Seq2Seq with Attention
- Bidirectional GRU architecture in the Encoder.
- Attention mechanism to improve focus and translation quality.
- GRU-based Decoder for generating output sequences.

## Loss Function
- CrossEntropyLoss function used across all models.
- Focus on sequences represented as a probability distribution over the vocabulary.
- Exclusion of padding in loss calculations.

## Optimization Algorithm
- Stochastic Gradient Descent (SGD) and AdamW Optimizer used for effective model training and generalization.

## Metrics and Experimental Results
- BLEU Score analysis for evaluating translation quality.
- BLEU Scores for English-Czech (23.1), English-German (22.6), and English-Ukrainian (22).

## Contributions and GitHub
- Sanjana Gadagoju: Data collection and preprocessing.
- Raakhal Rapolu: Model training and evaluation.
- Sushanth Grandhi: Model optimization, documentation, and report writing.

GitHub Repository: [Language Translation Transformer](https://github.com/raakhalrapolu/languagetranslation_transformer.git)

## References
1. [IEEE Document 9401982](https://ieeexplore.ieee.org/document/9401982)
2. [IEEE Document 9544509](https://ieeexplore.ieee.org/document/9544509)
3. [ArXiv 1912.07274](https://arxiv.org/pdf/1912.07274.pdf)
4. [TensorFlow Transformer Tutorial](https://www.tensorflow.org/text/tutorials/transformer)
5. [ArXiv 1808.03867](https://arxiv.org/abs/1808.03867)
