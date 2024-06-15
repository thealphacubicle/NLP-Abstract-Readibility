import aiohttp
import xml.etree.ElementTree as ET
import pandas as pd
import textstat as ts
import urllib.parse

async def fetch_arxiv_data(query, start=0, max_results=10):
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
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

def parse_arxiv_data(data):
    """
    Parses the XML data from ArXiv and extracts relevant information.

    Args:
        data (str): The raw XML data.

    Returns:
        list: A list of dictionaries containing paper details.
    """
    # XML processing code provided by ChatGPT
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

async def get_abstracts(queries, max_results):
    """
    Retrieves the abstracts of papers from ArXiv based on the given query.

    Args:
        queries (str or list): List of search queries or string for a single query
        max_results (int): Maximum number of results to fetch.

    Returns:
        list: A list of abstracts.
        queries (list): A list of queries matching the number of abstracts fetched.
    """
    if isinstance(queries, str):
        queries = [queries]

    abstracts = []
    lens = {query: 0 for query in queries}
    for query in queries:
        data = await fetch_arxiv_data(query, max_results=max_results)
        papers = parse_arxiv_data(data)
        lens[query] = len(papers)
        abstracts.extend([paper['summary'] for paper in papers])

    # Redefine the queries list to match the number of abstracts fetched
    queries = [query for query in queries for _ in range(lens[query])]

    return queries, abstracts

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
    }

def batch_add_to_df(abstracts, queries):
    """
    Adds the given data to the DataFrame.

    Args:
        abstracts (list): The list of abstract strings.
        queries_list (str or list): List of queries or string of singular query.

    Returns:
        pandas.DataFrame: The DataFrame with added data.
    """

    # Check to make sure length of abstracts and queries are the same
    assert len(abstracts) == len(queries), (("Length of abstracts and queries should be the same."
                                            "Length of abstracts: {}, Length of queries: {}")
                                            .format(len(abstracts), len(queries)))

    df_list = []
    for query, abstract in zip(queries, abstracts):
        df_list.append({'query': query, 'abstract': abstract})
        df_list[-1].update(get_text_scores(abstract))
    df = pd.DataFrame(df_list)
    return df