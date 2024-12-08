# **Personalized News Recommendation System Using Reinforcement Learning**

This project implements a personalized news recommendation system using Reinforcement Learning (RL). By leveraging the Proximal Policy Optimization (PPO) algorithm and a custom Gymnasium environment, the system optimizes Click-Through Rate (CTR) based on user preferences and simulated article metadata.

---

## **Features**
- **Custom Environment**: A Gymnasium-compatible environment simulates user interactions with news articles.
- **Reinforcement Learning**: Uses the PPO algorithm from Stable-Baselines3 for training.
- **MIND Dataset**: Employs a subset of the MIND-small dataset for efficient computation.
- **Evaluation**: Compares RL agent performance with a random policy.

---

## **Directory Structure**

```
project_root/
├── MINDsmall_train/
│   ├── news.tsv           # Metadata for news articles
│   ├── behaviors.tsv      # User interaction data
├── data_simulation.py      # Script for loading dataset and simulating articles/users
├── news_rec_env.py         # Custom Gymnasium environment
├── train_agent.py          # Script to train the RL agent
├── evaluate_agent.py       # Script to evaluate the RL agent
├── requirements.txt        # Required Python libraries
```

---

## **Getting Started**

### **1. Prerequisites**
Ensure you have the following installed:
- **Python**: Version 3.10 or later.
- **Virtual Environment Tool**: Either `venv` or `conda`.

---

### **2. Clone the Repository**
```bash
git clone https://github.com/Farabi667/AI-Term-Project.git
cd AI-Term-Project
```

---

### **3. Setting Up the Environment**
#### **Using `venv` (Linux/macOS)**:
```bash
python3.10 -m venv rl_env
source rl_env/bin/activate
```

#### **Using `conda` (Linux/macOS)**:
```bash
conda create -n rl_env python=3.10
conda activate rl_env
```

#### **Using `venv` (Windows)**:
```cmd
python -m venv rl_env
rl_env\Scripts\activate
```

#### **Using `conda` (Windows)**:
```cmd
conda create -n rl_env python=3.10
conda activate rl_env
```

---

**Note**: After activating the virtual environment, you must restart your terminal for the changes to take effect. This ensures proper environment isolation.

---

### **4. Install Dependencies**
Install the required Python packages:
```bash
pip install -r requirements.txt
```

---

## **Dataset Details**
We use a subset of **MIND-small training dataset** (`MINDsmall_train/`) to simulate articles and user interactions. This subset is chosen because the full dataset requires substantial computational power, making it impractical for smaller setups.

- **`news.tsv`**: Contains metadata for news articles (e.g., categories, subcategories, titles).
- **`behaviors.tsv`**: Contains user interaction data, including browsing and click logs.

---

## **Running the Project**

### **Training the RL Agent**
Train the agent with the following command:
```bash
python train_agent.py --timesteps 20000 --n_envs 2
```
- **`--timesteps`**: Number of training steps. Increase for better results.
- **`--n_envs`**: Number of parallel environments for training.

### **Evaluating the RL Agent**
Evaluate the trained agent:
```bash
python evaluate_agent.py --num_users 20 --num_episodes 100
```
- **`--num_users`**: Number of test users to simulate.
- **`--num_episodes`**: Number of evaluation episodes.

---

## **How the Code Works**

### **1. Data Simulation**
- **Script**: `data_simulation.py`
- **Purpose**: Loads the `news.tsv` and `behaviors.tsv` files and simulates:
  - Articles with topics and quality scores.
  - User profiles with topic preference vectors.

### **2. Training the RL Agent**
- **Script**: `train_agent.py`
- **Purpose**: Trains an RL agent using PPO to recommend articles that maximize CTR.

### **3. Evaluating the RL Agent**
- **Script**: `evaluate_agent.py`
- **Purpose**: Tests the trained agent against a random policy and computes CTR.

---

## **Tuning Parameters for Better Results**

### **Training**
- **Increase Timesteps**: Allows the agent to explore more and refine its policy.
  ```bash
  python train_agent.py --timesteps 50000
  ```
- **Increase Parallel Environments**: Improves training efficiency.
  ```bash
  python train_agent.py --n_envs 4
  ```

### **Evaluation**
- **Increase Test Users**: Tests generalization across a larger population.
  ```bash
  python evaluate_agent.py --num_users 50
  ```
- **Increase Episodes**: Produces more reliable CTR estimates.
  ```bash
  python evaluate_agent.py --num_episodes 200
  ```

---

## **Deliverables**
- **Source Code**: Full implementation available in this repository.
- **Documentation**: User manual and presentation slides in the `docs` directory.
- **Video Demo**: Watch the project walkthrough on YouTube:
  - **[Demo Video Link](https://youtu.be/EZZ9jvvYmPs?si=uibTQaU5dc3TmWt5)**.

---

## **FAQ**

### **Q1: What dataset is used?**
We use a subset of the MIND-small training dataset (`MINDsmall_train/`) to simulate articles and user interactions. This subset reduces computational overhead while retaining usability.

### **Q2: How do I customize the project?**
- Modify `data_simulation.py` to create different user behavior patterns or article distributions.
- Adjust training hyperparameters in `train_agent.py` for experimentation.

### **Q3: How do I troubleshoot training issues?**
- **Serialization Errors**: Update Python to version 3.10 or later.
- **Performance Plateaus**: Increase timesteps or adjust learning rates.

---

## **Future Enhancements**
- Real-time data integration for live recommendations.
- Advanced reward mechanisms for diverse user preferences.
- Deployment on a scalable platform for production use.

---

## **License**
This project is licensed under the MIT License. See the `LICENSE` file for details.
