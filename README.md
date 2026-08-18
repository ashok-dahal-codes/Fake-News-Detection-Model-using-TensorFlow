# Fake News Detection using TensorFlow

## Project Overview

This project uses a deep learning model to classify news articles as **REAL** or **FAKE**.

The model uses the **title of a news article** as input and combines **GloVe word embeddings**, **Convolutional Neural Networks (CNN)**, and **Long Short-Term Memory (LSTM)** networks to perform binary classification.

## Dataset

The project uses a `news.csv` dataset containing the following columns:

* `title` — News article title
* `text` — News article content
* `label` — Classification label (`REAL` or `FAKE`)

The `Unnamed: 0` column is removed because it is only an index column.

## Technologies Used

* Python
* Pandas
* NumPy
* TensorFlow / Keras
* Scikit-learn
* GloVe Word Embeddings

## Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the dataset using Pandas.
2. Removed the unnecessary `Unnamed: 0` column.
3. Converted `REAL` and `FAKE` labels into numerical values using `LabelEncoder`.
4. Selected the first 3,000 news articles for the experiment.
5. Tokenized the news titles using Keras `Tokenizer`.
6. Converted words into numerical sequences.
7. Applied padding to make the input sequences compatible with the neural network.
8. Split the data into training and testing portions.

## GloVe Word Embeddings

The project uses the **50-dimensional GloVe embeddings** provided by Stanford NLP.

The embeddings are used to initialize the model's embedding layer. The embedding weights are kept frozen during training.

The GloVe file used is:

```text
glove.6B.50d.txt
```

## Model Architecture

The neural network consists of the following layers:

```text
Input
  |
Embedding
  |
Dropout
  |
Conv1D
  |
MaxPooling1D
  |
LSTM
  |
Dense
  |
Output
```

### Model Configuration

* Embedding dimension: `50`
* Convolution filters: `64`
* Kernel size: `5`
* Max pooling size: `4`
* LSTM units: `64`
* Output activation: `sigmoid`
* Optimizer: `Adam`
* Loss function: `binary_crossentropy`
* Epochs: `50`

## Training

The model was trained using 3,000 news samples, with approximately 10% used as the validation/testing portion.

Training accuracy increased substantially during training, reaching approximately:

```text
Training Accuracy: 98.26%
Validation Accuracy: 77.33%
```

However, the difference between training and validation accuracy indicates **significant overfitting**. The model performs very well on the training data but generalizes considerably less effectively to unseen data.

The validation loss also increases substantially during later epochs, which further supports the presence of overfitting.

## Prediction

The trained model can classify a new news headline.

Example:

```text
"Karry to go to France in gesture of sympathy"
```

The model predicted:

```text
This news is False
```

## Project Structure

A recommended project structure is:

```text
Fake-News-Detection/
│
├── news.csv
├── model.h5
├── glove.6B/
│   └── glove.6B.50d.txt
├── fake_news_detection.ipynb
├── requirements.txt
└── README.md
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Install the required libraries:

```bash
pip install numpy pandas tensorflow scikit-learn
```

## Running the Project

1. Clone or download the project.
2. Create and activate the virtual environment.
3. Install the required dependencies.
4. Place `news.csv` in the project directory.
5. Download and extract the GloVe embeddings.
6. Run the Jupyter Notebook.
7. Train the model.
8. Use the prediction section to classify a new news headline.

## Model Performance

| Metric              | Result |
| ------------------- | -----: |
| Training Accuracy   | 98.26% |
| Validation Accuracy | 77.33% |
| Training Loss       | 0.0467 |
| Validation Loss     | 0.9286 |

The validation accuracy is the more important number for judging generalization. Therefore, the model should **not** be described as a 98% accurate fake-news detector.

## Limitations

* Only the **news title** is used by the trained model, despite the dataset containing article text.
* Only 3,000 samples were used for training.
* The model shows considerable overfitting.
* Validation accuracy is substantially lower than training accuracy.
* The dataset is relatively small for a neural-network-based NLP classification task.
* A single validation split was used rather than cross-validation or a separate held-out test set.
* The model should not be treated as a reliable real-world fact-checking system.

## Possible Improvements

The model could be improved by:

* Using the complete dataset instead of only 3,000 samples.
* Including both `title` and `text`.
* Increasing the maximum sequence length.
* Adding Early Stopping.
* Adding dropout and regularization.
* Using a proper train/validation/test split.
* Fine-tuning the embedding layer.
* Using pretrained transformer models such as BERT.
* Evaluating the model using precision, recall, F1-score, and a confusion matrix.

## Conclusion

This project demonstrates how deep learning can be applied to **fake news classification** using NLP techniques. GloVe embeddings provide pretrained word representations, while CNN and LSTM layers learn patterns in news headlines for binary classification.

The current model achieves **77.33% validation accuracy**, but the large gap between training and validation performance shows that overfitting is a major issue. Further preprocessing, more training data, regularization, and modern NLP architectures could improve its ability to generalize to unseen news.
