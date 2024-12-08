import pickle
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.env_util import make_vec_env
from data_simulation import load_mind_data, generate_articles_from_mind, generate_users_from_mind
from news_rec_env import NewsRecommendationEnv
import argparse
import os
import random
import numpy as np

def parse_args():
    """
    Parse command-line arguments.

    Returns:
    - args (argparse.Namespace): Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train News Recommendation Agent")
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level (0, 1, 2)')
    parser.add_argument('--timesteps', type=int, default=20000, help='Total training timesteps')
    parser.add_argument('--n_envs', type=int, default=2, help='Number of parallel environments')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()

def main():
    args = parse_args()

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load the MIND dataset
    print("Loading MIND dataset...")
    news_df, behaviors_df = load_mind_data()

    # Generate articles and users from MIND dataset
    print("Generating articles...")
    articles_df = generate_articles_from_mind(news_df)
    print("Generating users...")
    users = generate_users_from_mind(behaviors_df, news_df)

    # Sample a subset for quicker training
    articles_df = articles_df.sample(n=500, random_state=args.seed).reset_index(drop=True)
    users = random.sample(users, 100)

    print(f"Number of articles after sampling: {len(articles_df)}")
    print(f"Number of users after sampling: {len(users)}")

    articles_df.to_csv('articles_used.csv', index=False)

    # Create vectorized environments using env_kwargs
    print(f"Creating {args.n_envs} parallel environments...")

    env_kwargs = {'users': users, 'articles': articles_df}

    def make_env():
        return NewsRecommendationEnv(**env_kwargs)

    env = make_vec_env(
        make_env,
        n_envs=args.n_envs,
        seed=args.seed
    )

    eval_env = make_vec_env(
        make_env,
        n_envs=1,
        seed=args.seed
    )

    # After initializing the environment
    topic_to_idx = env.get_attr('topic_to_idx')[0]
    with open('topic_to_idx.pkl', 'wb') as f:
        pickle.dump(topic_to_idx, f)

    # Define the model with tuned hyperparameters
    print("Initializing PPO model...")
    model = PPO(
        'MlpPolicy',
        env,
        verbose=args.verbose,
        n_steps=512,
        batch_size=32,
        n_epochs=2,
        learning_rate=3e-4,
        policy_kwargs=dict(net_arch=[64, 32]),
        seed=args.seed
    )

    # Ensure the logs directory exists
    os.makedirs('./logs/best_model/', exist_ok=True)
    os.makedirs('./logs/', exist_ok=True)

    # Early stopping callback
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=3,
        min_evals=5,
        verbose=1
    )

    # Evaluation callback with reduced frequency
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./logs/best_model/',
        log_path='./logs/',
        eval_freq=2000,
        deterministic=True,
        render=False,
        callback_after_eval=stop_callback
    )

    # Train the agent with optimized timesteps
    print(f"Starting training for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps, callback=eval_callback)
    
    # Exclude the environment, logger, and callbacks when saving
    model._callback = None  # Clear the callback
    model.set_logger(None)  # Clear the logger
    model.save("news_recommendation_agent", exclude=["env", "logger", "_callback"])
    print("Training completed and model saved as 'news_recommendation_agent.zip'.")

if __name__ == '__main__':
    main()
