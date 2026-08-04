# ArXiv Paper Classification
A multi-label classification project that classifies scientific papers into multiple arXiv categories (e.g., `cs.CV`, `cs.LG`, `stat.ML`) based on their text summaries. It leverages pre-trained sentence transformers for text embeddings and a One-vs-Rest Random Forest classifier for multi-label prediction.

## Project Structure

```
├── abstract_classification.py    # Main classification pipeline with sentence embeddings
├── arxiv_data.csv                # Dataset containing arXiv papers
└── README.md                     # Project documentation
```

## Features

### Multi-Label Classification
Unlike single-label classification, each paper can belong to **multiple categories** simultaneously (e.g., a paper can be both `cs.CV` and `cs.LG`).

### GPU Acceleration
Automatically detects and utilizes GPU if available via PyTorch CUDA support.

### Sentence Embeddings
Uses [`google/embeddinggemma-300m`](https://huggingface.co/google/embeddinggemma-300m) from Sentence Transformers to convert text summaries into dense vector representations.

### One-vs-Rest Strategy
Trains one binary Random Forest classifier per category, enabling multi-label predictions.

## Dataset

The `arxiv_data.csv` file contains:
- **summaries**: Text abstracts/summaries of arXiv papers
- **terms**: List of arXiv category labels (e.g., `['cs.CV', 'cs.LG']`)

Data is sampled to 20,000 records with an 80/20 train/test split.

## Usage

### Run the classifier
```bash
python abstract_classification.py
```

## Model Architecture

```
Text Summary → Sentence Embedding (300M params) → OneVsRest(RandomForest) → Multi-label Categories
```

## Evaluation Metrics

- **F1-Score (samples)**: Primary metric for multi-label problems, averaged across all samples
- **Accuracy**: Exact match ratio (all labels must be correct)

Sample output:
```
Test F1 (samples): 0.XXXX
Test Accuracy: 0.XXXX
Sample predictions: [['cs.CV', 'cs.LG'], ['stat.ML'], ...]
```

## Key Implementation Details

### Label Parsing
Categories are stored as stringified Python lists (e.g., `"['cs.CV', 'cs.LG']"`), parsed using `ast.literal_eval()`.

### MultiLabelBinarizer
Converts list-of-lists labels into binary matrix format for scikit-learn compatibility:
- `fit_transform` on training labels
- `transform` (no refit) on test labels

### Embedding Strategy
- Train and test summaries are encoded **after** the split to prevent data leakage
- Embeddings are computed in batch for efficiency
