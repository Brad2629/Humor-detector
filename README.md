# 🤖 Humor Detection with Deep Learning

This project explores the use of deep learning to classify short text snippets as humorous or non-humorous. Using a dataset of 200,000 labeled texts from Kaggle, I trained a bidirectional LSTM model with word embeddings to detect humor in sentences. The final model achieves an accuracy of **93.4%** on the validation set.

## 📂 Dataset

- **Source**: [200K Short Texts for Humor Detection](https://www.kaggle.com/datasets/deepcontractor/200k-short-texts-for-humor-detection)
- **Size**: 200,000 text samples equally split between humorous and non-humorous

## 🧠 Model Architecture

- Tokenization and padding with Keras
- Custom embedding matrix with 250 dimensions
- Bidirectional LSTM + TimeDistributed layer
- Final Dense layers with softmax activation
- Loss: Sparse Categorical Crossentropy
- Optimizer: Adam

## 📈 Performance

- **Validation Accuracy**: 93.4%
- **Precision/Recall/F1**:
  - Humorous: 0.94 / 0.93 / 0.934
  - Non-Humorous: 0.93 / 0.94 / 0.935

## 📊 Visualizations

- Pie chart showing balanced dataset
- Histograms of text length
- Word clouds for humorous vs. non-humorous text
- Confusion matrix heatmap
- Accuracy/loss graphs across 20 training epochs

## 🧪 Sample Predictions

Example input sentences and their predicted humor classification:

```plaintext
"I told my wife she was drawing her eyebrows too high; she looked surprised."
→ Humor detected: True

"The cat slept peacefully on the warm windowsill."
→ Humor detected: False
```

## 🔮 Future Work

- Explore transformer-based models (e.g., BERT)
- Integrate part-of-speech tagging or semantic analysis
- Improve prediction confidence and filtering
