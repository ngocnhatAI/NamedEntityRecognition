# Named Entity Recognition (NER) using RNNs & CNNs

This repository provides a PyTorch implementation of Named Entity Recognition (NER) using six deep learning models, including RNNs, CNNs, and their bidirectional variants. We enhance performance with GloVe embeddings for better word representation and feature extraction.

# Models Implemented

We explore the effectiveness of different architectures for the NER task:
- GRU - Gated Recurrent Unit, a simpler alternative to LSTM.
- LSTM - Long Short-Term Memory, effective for long-range dependencies.
- BiGRU - Bidirectional GRU, capturing context from both directions.
- BiLSTM - Bidirectional LSTM, improving context understanding.
- CNN + BiGRU - Using CNN for feature extraction before passing to BiGRU.
- CNN + BiLSTM - Combining CNN with BiLSTM for optimal performance.

# Result 

| Model       | Dev Set Precision | Dev Set Recall | Dev Set F1 | Test Set Precision | Test Set Recall | Test Set F1 |
|-------------|-------------------|----------------|-------------|--------------------|-----------------|--------------|
| GRU         | 0.87              | 0.69           | 0.74        | 0.83               | 0.69            | 0.74         |
| LSTM        | 0.87              | 0.81           | 0.82        | 0.82               | 0.77            | 0.79         |
| BGRU        | 0.88              | 0.84           | 0.86        | 0.82               | 0.82            | 0.82         |
| BLSTM       | 0.91              | 0.83           | 0.87        | 0.86               | 0.81            | 0.84         |
| BGRU-CNN    | 0.92              | 0.89           | 0.91        | 0.86               | 0.88            | 0.87         |
| BLSTM-CNN   | **0.93**        | **0.92**       | **0.93**    | **0.87**           | **0.89**        | **0.88**     |

# Dataset
###  conll-2003 
