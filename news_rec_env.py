import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class NewsRecommendationEnv(gym.Env):
    """
    Custom Gymnasium environment for news recommendation.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, users, articles, topic_to_idx=None):
        super(NewsRecommendationEnv, self).__init__()
        self.users = users
        self.articles = articles.reset_index(drop=True)
        self.num_articles = len(articles)

        if topic_to_idx is None:
            self.unique_topics = articles['topic'].unique()
            self.num_topics = len(self.unique_topics)
            self.topic_to_idx = {topic: idx for idx, topic in enumerate(self.unique_topics)}
        else:
            self.topic_to_idx = topic_to_idx
            self.unique_topics = list(self.topic_to_idx.keys())
            self.num_topics = len(self.unique_topics)

        self.idx_to_topic = {idx: topic for topic, idx in self.topic_to_idx.items()}

        self.action_space = spaces.Discrete(self.num_articles)
        # Observation: User preferences vector
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_topics,), dtype=np.float32)

        self.current_user = None
        self.current_user_preferences = None

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment by selecting a new user.
        """
        super().reset(seed=seed)
        # Randomly select a user
        self.current_user = random.choice(self.users)
        # Create the preference vector in the order of topics
        self.current_user_preferences = np.array([
            self.current_user['preferences'].get(self.idx_to_topic[i], 0.0)
            for i in range(self.num_topics)
        ], dtype=np.float32)
        info = {}
        return self.current_user_preferences, info

    def step(self, action):
        """
        Take an action (recommend an article) and return the outcome.
        """
        terminated = True  # One step per episode
        truncated = False

        # Access article efficiently
        article = self.articles.iloc[action]
        topic = article['topic']
        quality = article['quality']
        user_pref = self.current_user['preferences'].get(topic, 0.0)
        # Probability of click depends on user preference and article quality
        click_prob = user_pref * quality
        reward = click_prob  # Use click probability as reward

        info = {'article_id': article['article_id'], 'click_prob': click_prob}
        return self.current_user_preferences, reward, terminated, truncated, info

    def render(self):
        """
        Render the environment. Not implemented.
        """
        pass
