---

# Hate Speech Detection using LSTM

This project demonstrates the use of a deep learning approach (LSTM) to classify tweets as **hate speech**, **offensive language**, or **neither**. The model is trained on a dataset sourced from Kaggle.

---

## 📌 Dataset

**Source:** [Kaggle - Hate Speech and Offensive Language Dataset](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset/data)  
**File Used:** `labeled_data.csv`  
**Shape:** 24,783 rows × 7 columns

---

## 🛠️ Preprocessing Pipeline

1. **Dropped Irrelevant Columns**  
   Columns like `Unnamed: 0`, `count`, and individual class scores were removed.
   
2. **Text Cleaning**
   - Removed special characters, emojis, and numeric values
   - Reduced multiple spaces

3. **NLP Tasks**
   - Lemmatization using `spaCy`
   - Stopword removal

4. **Tokenization & Encoding**
   - One-hot encoding of cleaned text
   - Sequence padding for uniform input size

---

## ⚖️ Class Imbalance Handling

Used **SMOTE (Synthetic Minority Over-sampling Technique)** to address class imbalance:

| Class | Description         | Count (Before SMOTE) |
|-------|---------------------|----------------------|
| 0     | Hate Speech         | 1,430                |
| 1     | Offensive Language  | 19,190               |
| 2     | Neither             | 4,163                |

---

## 🧠 Model Architecture

Built using **TensorFlow** and **Keras**:

- **Embedding Layer**: Converts words to dense vectors
- **LSTM Layers**: Three stacked LSTM layers for sequential pattern learning
- **Dense Output Layer**: 3 neurons with `softmax` activation for multi-class classification

```python
model = keras.Sequential([
    keras.layers.Embedding(vocab_size, dimension),
    keras.layers.LSTM(100, return_sequences=True),
    keras.layers.LSTM(50, return_sequences=True),
    keras.layers.LSTM(50),
    keras.layers.Dense(3, activation='softmax')
])
```

---

## ⚙️ Compilation & Training

- **Loss Function**: `sparse_categorical_crossentropy`
- **Optimizer**: `Adam`
- **Epochs**: 10
- **Batch Size**: 32

### Training Accuracy Over Epochs:

| Epoch | Accuracy | Loss  |
|-------|----------|-------|
| 1     | 77.35%   | 0.5083|
| 5     | 97.95%   | 0.0712|
| 10    | 99.33%   | 0.0202|

---

## 🧪 Evaluation

- **Train/Test Split**: 80/20
- **Metrics Used**:
  - Classification Report
  - Confusion Matrix
  - Accuracy

---

## 📊 Libraries Used

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `spacy` for NLP tasks
- `tensorflow.keras` for deep learning
- `sklearn` for preprocessing and metrics
- `imblearn` for SMOTE

---

## 🔮 Future Enhancements

- Use of BERT or other transformer-based models
- Add visualization for attention mechanism
- Hyperparameter tuning and early stopping

---

## 📁 Project Structure

```
hate-speech-lstm/
│
├── labeled_data.csv
├── hate_speech_detection.ipynb
├── README.md
```

---

## 🤝 Acknowledgements

- [Davidson et al., Hate Speech and Offensive Language Dataset](https://github.com/t-davidson/hate-speech-and-offensive-language)

---

