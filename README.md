# Yelp Sentiment Analysis

Sentiment classification of Yelp reviews using ensemble learning with transformer-based models (RoBERTa and DeBERTa).

## Problem

Analyzing customer sentiment from Yelp reviews is essential for businesses to understand customer satisfaction, identify areas for improvement, and address feedback effectively. This project classifies Yelp reviews into sentiment categories to help businesses gain actionable insights from customer feedback.

## Data Source

This project uses the [Yelp Review Dataset](https://huggingface.co/datasets/Yelp/yelp_review_full) from Hugging Face, originally curated by Xiang Zhang during the Yelp Dataset Challenge 2015. The dataset contains:
- **650,000 training samples** (130,000 per star rating)
- **50,000 test samples** (10,000 per star rating)
- Reviews labeled from 1 to 5 stars

### Getting the Data

1. The dataset can be loaded directly from Hugging Face:
   ```python
   from datasets import load_dataset
   dataset = load_dataset('yelp_review_full')
   ```

2. For manual download, visit the [Hugging Face dataset page](https://huggingface.co/datasets/Yelp/yelp_review_full)

3. Place any downloaded data files in a `Data/` directory (this directory is git-ignored)

## Approach

### Classification Schemes

The project explores three classification approaches:
- **5-class**: Original 1-5 star ratings
- **3-class**: Negative (1-2 stars), Mixed (3 stars), Positive (4-5 stars)
- **2-class**: Binary negative/positive sentiment

### Models

1. **Baseline**: Bag-of-Words with Naive Bayes classifier
2. **Transformer Models**:
   - RoBERTa-base (sensitive to extreme sentiments)
   - DeBERTa-base (captures subtle sentiment variations)
3. **Ensemble Learning**: Combines softmax outputs from both transformers using:
   - Averaging
   - Random Forest
   - Multi-Layer Perceptron (MLP)
   - Stacking, Voting, Decision Trees, Gradient Boosting

### Data Processing

- Filtered reviews to 100-200 word count range
- Balanced sampling across all classes
- Tokenization using model-specific tokenizers (BertTokenizer, DebertaTokenizer, RobertaTokenizer)

## Key Results

### Baseline (Naive Bayes)
- **2-class**: 88% accuracy
- **3-class**: 68% accuracy
- **5-class**: 52% accuracy

### Transformer Models (Individual)
- **RoBERTa**: 54.8% test accuracy on 5-class, strong on extreme sentiments
- **DeBERTa**: 57% test accuracy on 5-class, balanced across classes
- **BERTweet (fine-tuned)**: 71% accuracy, 72% precision, 69% F1

### Ensemble Models (Final)
- **3-class**: **79% Macro F1 and Micro F1** (11% improvement over baseline)
- **5-class**: **62% Macro F1 and Micro F1** (15% improvement over baseline)

The ensemble approach successfully combines RoBERTa's strength in identifying extreme sentiments with DeBERTa's ability to capture subtle differences, achieving significantly better performance than individual models.

## Repository Structure

```
├── Baseline_Model/          # Naive Bayes baseline implementation
│   ├── naive_bayes.ipynb
│   └── yelp_eda_naive_bayes.ipynb
├── EDA/                     # Exploratory data analysis
│   └── yelp_eda.ipynb
├── Ensemble/                # Ensemble learning experiments
│   └── Ensemble_Learn.ipynb
├── kev_bertweet/            # BERTweet fine-tuning experiments
│   ├── kev_finetune_model.ipynb
│   ├── tuned_bertweet.ipynb
│   └── huggingface_bertweet.ipynb
├── Model_Pipeline/          # Modular model pipeline
│   ├── src/
│   │   ├── models.py
│   │   ├── trainer.py
│   │   ├── utils.py
│   │   ├── main.py
│   │   └── datamodule.py
│   └── notebooks/
│       └── Notebook - Data Processing.ipynb
├── miscellaneous/           # Additional experiments
│   ├── Fine-tuned Models.ipynb
│   ├── 3 Class Model.ipynb
│   └── Subset Generation.ipynb
├── acl2023/                 # Research paper and figures
├── requirements.txt
├── LICENSE
└── README.md
```

## How to Run

### Prerequisites

Python 3.8+ required. Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the Models

1. **Baseline Model**:
   ```bash
   jupyter notebook Baseline_Model/naive_bayes.ipynb
   ```

2. **Ensemble Learning**:
   ```bash
   jupyter notebook Ensemble/Ensemble_Learn.ipynb
   ```

3. **BERTweet Fine-tuning**:
   ```bash
   jupyter notebook kev_bertweet/tuned_bertweet.ipynb
   ```

4. **Model Pipeline** (modular approach):
   ```bash
   cd Model_Pipeline
   python src/main.py
   ```

### Note on Notebooks

Some notebooks may contain large outputs. If you experience performance issues, clear outputs using:
```bash
jupyter nbconvert --clear-output --inplace <notebook>.ipynb
```

## Tech Stack

- **Deep Learning**: TensorFlow, Keras
- **Transformers**: Hugging Face Transformers (RoBERTa, DeBERTa, BERT)
- **ML Libraries**: scikit-learn, mlxtend
- **NLP**: NLTK, TextBlob
- **Data Processing**: pandas, numpy, datasets
- **Visualization**: matplotlib, seaborn, wordcloud

## Citation

If you use this work, please cite the original Yelp dataset:
> Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level Convolutional Networks for Text Classification. Advances in Neural Information Processing Systems.

## License

MIT License - see [LICENSE](LICENSE) file for details.

Copyright (c) 2024 Kevin Kuc

---

**Note**: The git history of this repository contains large data files that have been removed from the working tree. The `.gitignore` file now prevents these files from being committed in the future.
