# Humor Detection Using NLP and Deep Learning

This project focuses on detecting humor in short texts using both traditional preprocessing techniques and transformer-based models. The dataset consists of 200,000 labeled text samples from Kaggle, evenly split between humorous and non-humorous content.

## 📊 Dataset

- **Source:** [Kaggle - 200K Short Texts for Humor Detection](https://www.kaggle.com/datasets/deepcontractor/200k-short-texts-for-humor-detection)
- **Size:** 200,000 entries (100,000 humorous, 100,000 not humorous)
- **Format:** CSV with two main columns: `text` and `humor` (boolean)

## 🧠 Methods

The project explores two major approaches:

### 1. Traditional Deep Learning (Keras)
- **Preprocessing:** Tokenization, lowercasing, filtering of special characters
- **Embedding:** Keras Embedding layer
- **Model:** LSTM-based sequential neural network
- **Training:** Binary classification with accuracy and loss metrics

### 2. Transformer-Based Classification (BERT)
- **Tokenizer:** `BertTokenizer`
- **Model:** `BertForSequenceClassification` from HuggingFace Transformers
- **Training:** Using `Trainer` API with GPU acceleration
- **Evaluation:** Classification report and confusion matrix

## 📈 Results

- Clear distinction between humorous and non-humorous text
- BERT-based model significantly outperforms the LSTM model in accuracy and F1-score
- WordClouds and pie charts used for basic exploratory data analysis

## 🧰 Libraries Used

- Pandas, NumPy, Matplotlib, Seaborn
- TensorFlow/Keras
- HuggingFace Transformers
- Scikit-learn

## 🔍 Key Insights

- Humor detection is feasible with modern NLP tools
- Pretrained language models like BERT yield higher performance than traditional approaches
- Careful preprocessing and balanced data improve model reliability

## 👤 Author

**Bradley Paliska**  
`bp355` — Cornell University

