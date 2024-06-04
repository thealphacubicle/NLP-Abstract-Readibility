import urllib.request
import xml.etree.ElementTree as ET
import os
import pandas as pd
import textstat as ts

def fetch_arxiv_data(query, start=0, max_results=10):
    """
    Fetches data from ArXiv based on the given query.

    Args:
        query (str): The search query.
        start (int): The starting index of the results.
        max_results (int): Maximum number of results to fetch.

    Returns:
        str: The raw XML data fetched from ArXiv.
    """
    base_url = 'http://export.arxiv.org/api/query'
    query_params = {
        'search_query': query,
        'start': start,
        'max_results': max_results
    }
    url = f"{base_url}?{urllib.parse.urlencode(query_params)}"
    with urllib.request.urlopen(url) as response:
        return response.read().decode('utf-8')

def parse_arxiv_data(data):
    """
    Parses the XML data from ArXiv and extracts relevant information.

    Args:
        data (str): The raw XML data.

    Returns:
        list: A list of dictionaries containing paper details.
    """
    root = ET.fromstring(data)
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    entries = root.findall('atom:entry', ns)

    papers = []
    for entry in entries:
        paper = {
            'title': entry.find('atom:title', ns).text.strip(),
            'summary': entry.find('atom:summary', ns).text.strip(),
            'authors': [author.find('atom:name', ns).text.strip() for author in entry.findall('atom:author', ns)],
            'published': entry.find('atom:published', ns).text.strip(),
            'updated': entry.find('atom:updated', ns).text.strip(),
            'id': entry.find('atom:id', ns).text.strip()
        }
        papers.append(paper)

    return papers

def get_abstracts(query, max_results=10):
    """
    Retrieves the abstracts of papers from ArXiv based on the given query.

    Args:
        query (str): The search query.
        max_results (int): Maximum number of results to fetch.

    Returns:
        list: A list of abstracts.
    """
    data = fetch_arxiv_data(query, max_results=max_results)
    papers = parse_arxiv_data(data)
    abstracts = [paper['summary'] for paper in papers]


    yield from abstracts

def get_text_scores(text):
    """
    Returns the number of words, sentences, and characters in the given text.

    Args:
        text (str): The input text.

    Returns:
        dict: A dictionary containing the number of words, sentences, and characters.
    """
    words = text.split()
    sentences = text.split('.')
    characters = list(text)

    return {
        'num_words': len(words),
        'num_sentences': len(sentences),
        'num_characters': len(characters),
        'flesch_reading_ease': ts.flesch_reading_ease(text),
        'flesch_kincaid_grade': ts.flesch_kincaid_grade(text),
        'smog_index': ts.smog_index(text),
        'gunning_fog': ts.gunning_fog(text)
    }

def batch_add_to_df(abstracts, query):
    """
    Adds the given data to the DataFrame.

    Args:
        abstracts (list): The list of abstract strings.
        query (str): The search query.

    Returns:
        pandas.DataFrame: The DataFrame with added data.
    """
    df_list = []
    for i, abstract in enumerate(abstracts, 1):
        df_list.append({'query': query, 'abstract': abstract})
        df_list[-1].update(get_text_scores(abstract))
    df = pd.DataFrame(df_list)
    return df

def save_to_csv(df, filename):
    """
    Saves the DataFrame to a CSV file.

    Args:
        df (pandas.DataFrame): The DataFrame to be saved.
        filename (str): The name of the CSV file.
    """
    # Check if the file already exists. If it does, append the new df to the existing file.
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)

    else:
        df.to_csv(filename, index=False)



# Example usage
if __name__ == "__main__":
    query = "machine learning"
    abstracts = get_abstracts(query, max_results=100)
    # for i, abstract in enumerate(abstracts, 1):
    #     print(f"Abstract {i}:\n{abstract}\n")
    #     print("Abstract Scores: ", get_text_scores(abstract), "\n")

    df = batch_add_to_df(abstracts, query)
    save_to_csv(df, 'arxiv_data.csv')


