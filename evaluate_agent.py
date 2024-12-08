import numpy as np
import pandas as pd
import pickle
import argparse
import random
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from data_simulation import load_mind_data, generate_users_from_mind
from news_rec_env import NewsRecommendationEnv

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate News Recommendation Agent")
    parser.add_argument('--num_users', type=int, default=20, help='Number of test users')
    parser.add_argument('--num_episodes', type=int, default=100, help='Number of evaluation episodes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()

def evaluate_agent(env, model, num_episodes=100):
    total_rewards = []
    for _ in tqdm(range(num_episodes), desc="Evaluating Agent"):
        obs = env.reset()
        action, _ = model.predict(obs)
        obs, rewards, dones, infos = env.step(action)
        total_rewards.append(rewards[0])  # Extract the reward from the array
    ctr = np.mean(total_rewards)
    print(f"Agent CTR over {num_episodes} episodes: {ctr:.4f}")
    return ctr

def evaluate_random(env, num_episodes=100):
    total_rewards = []
    for _ in tqdm(range(num_episodes), desc="Evaluating Random Policy"):
        obs = env.reset()
        action = [env.action_space.sample()]  # Wrap action in a list
        obs, rewards, dones, infos = env.step(action)
        total_rewards.append(rewards[0])  # Extract the reward from the array
    ctr = np.mean(total_rewards)
    print(f"Random policy CTR over {num_episodes} episodes: {ctr:.4f}")
    return ctr

def main():
    args = parse_args()

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load the MIND dataset
    print("Loading MIND dataset...")
    news_df, behaviors_df = load_mind_data()
    users = generate_users_from_mind(behaviors_df, news_df)
    
    # Load the saved articles
    articles_df = pd.read_csv('articles_used.csv')

    # Sample test users
    if args.num_users > len(users):
        print(f"Requested {args.num_users} users, but only {len(users)} available. Using all users.")
        test_users = users
    else:
        test_users = random.sample(users, args.num_users)

    # Create the evaluation environment
    print(f"Creating evaluation environment with {len(test_users)} users...")

    # Load the topic_to_idx mapping
    with open('topic_to_idx.pkl', 'rb') as f:
        topic_to_idx = pickle.load(f)

    env_kwargs = {
        'users': test_users,
        'articles': articles_df,
        'topic_to_idx': topic_to_idx
    }

    def make_env():
        return NewsRecommendationEnv(**env_kwargs)

    env = make_vec_env(
        make_env,
        n_envs=1,  # Keep n_envs=1 for evaluation
        seed=args.seed
    )

    # Load the trained model and set the environment
    print("Loading the trained PPO model...")
    model = PPO.load("news_recommendation_agent", env=env)

    # Optionally, set a logger if needed
    new_logger = configure("./logs/", ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    # Evaluate the agent
    print(f"Evaluating the agent over {args.num_episodes} episodes...")
    agent_ctr = evaluate_agent(env, model, num_episodes=args.num_episodes)

    # Evaluate random policy
    print(f"Evaluating the random policy over {args.num_episodes} episodes...")
    random_ctr = evaluate_random(env, num_episodes=args.num_episodes)

    # Compare results
    improvement = (agent_ctr - random_ctr) * 100
    print(f"CTR Improvement: {improvement:.2f}%")

if __name__ == '__main__':
    main()
