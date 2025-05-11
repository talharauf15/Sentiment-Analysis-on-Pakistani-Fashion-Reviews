# Sentiment Analysis on Pakistani Fashion Reviews

This project aims to perform **sentiment analysis** on reviews related to the Pakistani fashion industry. Using a custom-labeled dataset, we clean and preprocess text data, visualize trends, extract features using **TF-IDF**, and apply machine learning models to classify sentiments as **Positive**, **Negative**, or **Neutral**.

## 📁 Dataset

* **File**: `PakistanBrandsFashionSentimentDataset.csv`
* **Attributes**:

  * `Text`: Review content
  * `Sentiment`: Sentiment label (Positive/Negative/Neutral)

## 📒 Jupyter Notebook

* **File**: `SentimentAnalysis.ipynb`
* Contains all steps from data loading to model evaluation.

## 🔍 Project Features

* Text preprocessing:

  * Lowercasing, punctuation & stopword removal
  * Tokenization using NLTK
* Exploratory Data Analysis (EDA):

  * Sentiment distribution
  * Most frequent words by sentiment
* Feature extraction:

  * **TF-IDF Vectorization**
* Sentiment classification using:

  * **Multinomial Naive Bayes**
  * **Support Vector Machine (SVM)**
  * **Logistic Regression**
* Evaluation Metrics:

  * Accuracy
  * Classification Report (Precision, Recall, F1-score)
  * Confusion Matrix

## 🛠️ Technologies & Tools Used

| Category           | Tools/Libraries                |
| ------------------ | ------------------------------ |
| Programming        | Python 3                       |
| IDE                | Jupyter Notebook (ipynb)       |
| Data Handling      | Pandas, NumPy                  |
| Visualization      | Matplotlib, Seaborn            |
| Text Processing    | NLTK (stopwords, tokenization) |
| ML Algorithms      | Scikit-learn                   |
| Feature Extraction | TfidfVectorizer (Scikit-learn) |

## 🚀 Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/talharauf15/Sentiment-Analysis-on-Pakistani-Fashion-Reviews.git
   cd Sentiment-Analysis-on-Pakistani-Fashion-Reviews
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Launch the notebook:

   ```bash
   jupyter notebook SentimentAnalysis.ipynb
   ```

## ✅ Model Performance

* Best performing model: **Support Vector Machine (SVM)**
* Accuracy achieved: **82.5%**

## 📌 Use Cases

* Monitoring public sentiment about fashion brands
* Helping designers understand customer feedback
* Enhancing customer service response strategies
