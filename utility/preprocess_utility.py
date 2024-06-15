import pandas as pd
import polars as pl
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

# Download NLTK data
import nltk
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

# Preprocess class registration for polars DataFrame
@pl.api.register_expr_namespace('text_processing')
class TextProcessing:
    def __init__(self, expr):
        self._expr = expr

    def preprocess(self):
        """Apply the preprocess_text function."""
        return self._expr.apply(preprocess_text)

def preprocess_text(text):
    """Preprocess the text by converting to lowercase, tokenizing, and removing stopwords.

    Args:
        text (str): The input text.

    Returns:
        list: The list of preprocessed words.
    """

    # Case folding and tokenization with stop words removal
    words = word_tokenize(text.lower())
    return [word for word in words if word.isalnum() and word not in stop_words]

def train_word2vec_model(sentences, vector_size=100, window=5, sg=1, min_count=1, epochs=10):
    """Train a Word2Vec model using the provided tokenized sentences.
    Args:
        sentences (list): The tokenized sentences.
        vector_size (int): The dimensionality of the word vectors.
        window (int): The maximum distance between the current and predicted word within a sentence.
        sg (int): The training algorithm: 1 for skip-gram; otherwise CBOW.
        min_count (int): Ignores all words with a total frequency lower than this.
        epochs (int): The number of iterations over the corpus.

    Returns:
        gensim.models.Word2Vec: The trained Word2Vec model.
    """
    model = Word2Vec(sentences=sentences, vector_size=vector_size, window=window, sg=sg, min_count=min_count)
    model.train(sentences, total_examples=len(sentences), epochs=epochs)
    return model

def get_abstract_embedding(abstract, model):
    """Get the embedding for a given abstract using the Word2Vec model."

    Args:
        abstract (str): The input abstract.
        model (gensim.models.Word2Vec): The Word2Vec model.

    Returns:
        numpy.ndarray: The average embedding for the abstract.
    """
    # Tokenize the abstract and remove stopwords
    tokenized_abstract = [word for word in word_tokenize(abstract.lower()) if word.isalnum() and word not in stop_words]

    # Get the embeddings for the words in the abstract
    embeddings = [model.wv[word] for word in tokenized_abstract if word in model.wv]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(model.vector_size)

def polars_to_pandas(df):
    """Convert a Polars DataFrame to a Pandas DataFrame.

    Args:
        df (polars.DataFrame): The input Polars DataFrame.

    Returns:
        pandas.DataFrame: The converted Pandas DataFrame.
    """
    pandas_df = pd.DataFrame()
    for col in df.columns:
        pandas_df[col] = df[col].to_list()

    return pandas_df

def preprocess_dataframe(df):
    """Preprocess the DataFrame by applying all the steps.

    Args:
        df (pandas.DataFrame): The input DataFrame.

    Returns:
        pandas.DataFrame: The preprocessed DataFrame.
    """
    # Convert to Polars DataFrame
    polars_df = pl.from_pandas(df)

    # Create the lazy Polars DataFrame and apply preprocessing
    result_df = (
        polars_df.lazy()
        .with_columns(pl.col('abstract').text_processing.preprocess().alias('processed_abstract'))
    ).collect()

    # Get tokenized sentences
    tokenized_sentences = result_df['processed_abstract'].to_list()

    # Train the Word2Vec model
    model = train_word2vec_model(tokenized_sentences)

    # Store embeddings for each abstract
    result_df = result_df.with_columns([
        pl.col('abstract').apply(lambda x: get_abstract_embedding(x, model)).alias('embedding')
    ])

    # Convert back to Pandas DataFrame
    return polars_to_pandas(result_df)
