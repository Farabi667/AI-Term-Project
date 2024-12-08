import pandas as pd
import numpy as np
import os

# Paths to the dataset files
TRAIN_PATH = './MINDsmall_train'
DEV_PATH = './MINDsmall_dev'

def load_mind_data(news_path=os.path.join(TRAIN_PATH, 'news.tsv'), behaviors_path=os.path.join(TRAIN_PATH, 'behaviors.tsv')):
    """
    Load the MIND dataset from the specified paths.

    Parameters:
    - news_path (str): Path to the news.tsv file.
    - behaviors_path (str): Path to the behaviors.tsv file.

    Returns:
    - news_df (pd.DataFrame): DataFrame containing news data.
    - behaviors_df (pd.DataFrame): DataFrame containing behaviors data.
    """
    # Load news data with essential columns
    news_df = pd.read_csv(
        news_path,
        sep='\t',
        header=None,
        names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'],
        usecols=['news_id', 'category', 'subcategory']  # Limit to necessary columns
    )
    
    # Optimize data types
    news_df['category'] = news_df['category'].astype('category')
    news_df['subcategory'] = news_df['subcategory'].astype('category')
    
    # Load behaviors data with essential columns
    behaviors_df = pd.read_csv(
        behaviors_path,
        sep='\t',
        header=None,
        names=['impression_id', 'user_id', 'time', 'history', 'impressions'],
        usecols=['user_id', 'history']  # Limit to necessary columns
    )
    
    return news_df, behaviors_df

def generate_articles_from_mind(news_df):
    """
    Generate a DataFrame of articles from the news DataFrame.

    Parameters:
    - news_df (pd.DataFrame): DataFrame containing news data.

    Returns:
    - articles_df (pd.DataFrame): DataFrame containing processed articles.
    """
    articles = []
    for _, row in news_df.iterrows():
        article = {
            'article_id': row['news_id'],
            'topic': row['category'],
            'quality': np.random.uniform(0.5, 1.0)  # Random quality score for demonstration
        }
        articles.append(article)
    return pd.DataFrame(articles)

def generate_users_from_mind(behaviors_df, news_df):
    users = []
    # Mapping from news_id to category
    news_topic_dict = news_df.set_index('news_id')['category'].to_dict()
    unique_topics = news_df['category'].unique()
    for user_id, group in behaviors_df.groupby('user_id'):
        history = group['history'].iloc[0]
        if isinstance(history, str):
            history_articles = history.split(' ')
            history_topics = [news_topic_dict.get(article_id) for article_id in history_articles]
            history_topics = [topic for topic in history_topics if topic is not None]
        else:
            history_topics = []

        user_preferences = {}
        for topic in unique_topics:
            if topic in history_topics:
                user_preferences[topic] = np.random.uniform(0.7, 1.0)
            else:
                user_preferences[topic] = np.random.uniform(0.0, 0.3)

        user = {
            'user_id': user_id,
            'preferences': user_preferences
        }
        users.append(user)
    return users

if __name__ == '__main__':
    news_df, behaviors_df = load_mind_data()
    articles_df = generate_articles_from_mind(news_df)
    users = generate_users_from_mind(behaviors_df)
    print("Sample Articles:")
    print(articles_df.head())
    print("\nSample Users:")
    print(users[:2])
