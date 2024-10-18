### LN Project (2024/25)
### Ana Alfaiate 102903 | Inês Trigueiro 102902 | Raquel Coelho 102881


# Feature Engineering
## 1. Imports

import numpy as np
import pandas as pd
# scikit-learn
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# pip install nltk
# pip install matplotlib
import nltk
import re
from nltk.stem import PorterStemmer
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm

import random
import string

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import RandomizedSearchCV

import joblib
import os

# Download necessary nltk data
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.corpus import wordnet

random.seed(42)

################# TEST NO LABELS ######################

# Load the dataset
file_path_test = 'test_no_labels.txt'
# Define the column names based on the data format description
columns = ['title', 'from', 'genre', 'director', 'plot']
test = pd.read_csv(file_path_test, delimiter= "\t", names=columns) # Check the separator
test_no_labels = test['director']
# Inspect the first few rows
print(test_no_labels.head())
print(test_no_labels.shape)

def remove_punctuation_from_texts(X):
    # Translation table to remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    
    # Remove punctuation from each text in X
    X_cleaned = [text.translate(translator) for text in X]
    
    return X_cleaned

def lowercase_texts(X):
    # Convert each text in X to lowercase
    X_lowercased = [text.lower() for text in X]
    
    return X_lowercased

def remove_stopwords_from_texts(X, stop_words=None):
        if stop_words is None:
            print(stopwords.words('english'))
            stop_words = set(stopwords.words('english'))  # Load default NLTK stopwords
        
        # Remove stop words from each text in X
        X_cleaned = [' '.join([word for word in text.split() if word.lower() not in stop_words]) for text in X]
        
        return X_cleaned


test_no_labels = remove_punctuation_from_texts(test_no_labels)
#print(test_no_labels[:10])

test_no_labels = lowercase_texts(test_no_labels)
#print(test_no_labels[:10])

Stop_words = None
#Stop_words=["the","is","and"]
test_no_labels = remove_stopwords_from_texts(test_no_labels, Stop_words)
print(test_no_labels[:10])

# Initialization
max_features= None
#max_features=50
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,3), min_df = 2, max_features=max_features)


model_filename = 'svm_model.pkl'
if os.path.exists(model_filename):
    # Load model
    clf = joblib.load('svm_model.pkl') 
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')



else:
    ## 2. Dataset load

    # Load the dataset
    file_path = 'train.txt'
    data = pd.read_csv(file_path, delimiter= "\t", names=columns) # Check the separator

    # Inspect the first few rows
    print(data.head())
    print(data.shape)

    # Pre-processing dataset
    X_inicial = data['plot']  # Check if the appropriate collumn name is 'plot'

    y = data['genre']      # Check if the appropriate collumn name is 'genre'
    labels = np.unique(y).tolist()


    ## 3. Pre-processing

    ### 3.1 Punctuation
    y_process = y 
    X_process = remove_punctuation_from_texts(X_inicial)
    #print(X_process[:10])


    ### 3.2 Lowercasing
    X_inicial = X_process
    y = y_process
    X_process = lowercase_texts(X_inicial)
    #print(X_process[:10])


    ### 3.3 Stop Words
    X_process = remove_stopwords_from_texts(X_inicial, Stop_words)
    y_process = y
    #print(X_process[:10])
    print(f"Size of the data set: {len(y_process)}")


    ## 4. Train/test split
    X_train = X_process
    y_train = y_process
    #X_train, X_test, y_train, y_test = train_test_split(X_process, y_process, test_size=0.2, random_state=42)
    #x_test_raw = X_test


    ## 5. Create vectors

    ### 5.1 TF-IDF 
    # Fit and transform the text data
    X_train = tfidf_vectorizer.fit_transform(X_train)
    #X_test = tfidf_vectorizer.transform(X_test)
    #print(X_train)
    print("Size of the data set (train): ", X_train.shape)
    #print(X_test)
    #print("Size of the data set (validation): ", X_test.shape)

    joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')


    ## 6. Classifiers
    ### Support Vector Machines (SVM)

    # === Train SVM model ===
    print("==== SVM ====")

    # Define the SVM model
    clf = svm.SVC(kernel='linear') # kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}

    # Fit the model
    clf.fit(X_train, y_train)

    '''# Predict using the test set
    x_pred = clf.predict(X_test)

    # List of genres (labels)
    labels = np.unique(y_test)
    '''

    '''# === Classification Report ===
    # Print overall accuracy, precision, recall, and f1-score by genre
    print("\nClassification Report:")
    report = classification_report(y_pred=x_pred, y_true=y_test, labels=labels, zero_division=1, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print(report_df)

    # === Overall Model Metrics ===
    overall_accuracy = accuracy_score(y_test, x_pred)
    overall_precision = report_df.loc['weighted avg']['precision']
    overall_recall = report_df.loc['weighted avg']['recall']
    overall_f1 = report_df.loc['weighted avg']['f1-score']

    print("\nOverall Model Performance:")
    print(f"Accuracy: {overall_accuracy:.2%}")
    print(f"Precision (weighted avg): {overall_precision:.2%}")
    print(f"Recall (weighted avg): {overall_recall:.2%}")
    print(f"F1-score (weighted avg): {overall_f1:.2%}")


    # === Confusion Matrix ===
    print("\nConfusion Matrix:")
    conf_matrix = confusion_matrix(y_test, x_pred, labels=labels)
    print(pd.DataFrame(conf_matrix, index=labels, columns=labels))


    # Assuming y_test and x_pred are defined
    if isinstance(y_test, pd.Series):
        y_test_array = y_test.to_numpy()  # Convert to NumPy array if it's a Series
    else:
        y_test_array = y_test  # If it's already a NumPy array

    # Calculate accuracy for each genre
    for genre in labels:
        genre_idx = np.where(y_test_array == genre)[0]  # Indices for this genre in y_test
        genre_accuracy = accuracy_score(y_test_array[genre_idx], x_pred[genre_idx])  # Calculate accuracy for this genre
        print(f"{genre}: {genre_accuracy:.2%}")
    '''

    # Save the trained model as a file
    joblib.dump(clf, model_filename)  
    print("Modelo SVM gravado com sucesso!")

test_no_labels = tfidf_vectorizer.transform(test_no_labels)
print("Size of the test set: ", test_no_labels.shape)
predicted_labels = clf.predict(test_no_labels)

# file results.txt
with open('results.txt', 'w') as f:
    for i in range(len(predicted_labels)):
        if i < len(predicted_labels) - 1:
            f.write(f"{predicted_labels[i]}\n")  # Write the label with newline
        else:
            f.write(f"{predicted_labels[i]}")  # Last label without newline
        #print(test['director'][i])
        print(predicted_labels[i])