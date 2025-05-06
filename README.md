# Named Entity Recognition (NER) with RNNs, CNNs, and BERT

This repository provides a PyTorch-based implementation for Named Entity Recognition (NER) using a range of deep learning models, including RNNs, CNNs, and BERT. We leverage **GloVe word embeddings** for RNN/CNN models and **pretrained BERT** for contextual embeddings, allowing robust entity recognition from both local and global contexts.

## Models Implemented

We compare several architectures for the NER task:

* **GRU** – Gated Recurrent Unit, a lightweight alternative to LSTM.
* **LSTM** – Long Short-Term Memory, capturing long-range dependencies.
* **BiGRU** – Bidirectional GRU, leveraging context from both past and future tokens.
* **BiLSTM** – Bidirectional LSTM, enhancing contextual understanding.
* **CNN + BiGRU** – CNN extracts local features, passed to BiGRU for sequential modeling.
* **CNN + BiLSTM** – Combines local pattern extraction and bidirectional sequence modeling.
* **BERT** – Pretrained Transformer-based model, fine-tuned for NER with token-level classification.

## Results (Micro Average Metrics)

| Model        | Dev Set Micro Precision | Dev Set Micro Recall | Dev Set Micro F1 | Test Set Micro Precision | Test Set Micro Recall | Test Set Micro F1 |
| ------------ | ----------------------- | -------------------- | ---------------- | ------------------------ | --------------------- | ----------------- |
| GRU          | 0.87                    | 0.69                 | 0.74             | 0.83                     | 0.69                  | 0.74              |
| LSTM         | 0.87                    | 0.81                 | 0.82             | 0.82                     | 0.77                  | 0.79              |
| BiGRU        | 0.88                    | 0.84                 | 0.86             | 0.82                     | 0.82                  | 0.82              |
| BiLSTM       | 0.91                    | 0.83                 | 0.87             | 0.86                     | 0.81                  | 0.84              |
| CNN + BiGRU  | 0.92                    | 0.89                 | 0.91             | 0.86                     | 0.88                  | 0.87              |
| CNN + BiLSTM | 0.93                    | 0.92                 | 0.93             | 0.87                     | 0.89                  | 0.88              |
| **BERT**     | **0.94**                | **0.95**             | **0.95**         | **0.90**                 | **0.92**              | **0.91**          |

## Dataset

We use the standard **CoNLL-2003** dataset, which includes annotations for four entity types: `PER` (person), `ORG` (organization), `LOC` (location), and `MISC` (miscellaneous).

