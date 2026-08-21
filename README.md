# Kaggle Autonomous Agent Prediction

Link: https://www.kaggle.com/competitions/autonomous-agent-prediction-beta

This repository contains the configuration and logic for a fully autonomous machine learning agent designed to win tabular prediction competitions. 

## Architecture & Strategy

The agent mimics Cdeotte's classic hill climbing idea, focussing on extreme model diversity (LGBM, XGBoost, CatBoost, ExtraTrees, Logistic Regression) paired with feature engineering techniques like Nested Target Encoding, Frequency Encoding, Multi-scale Binning, Categorical Bigrams etc that were mentioned in one of his kaggle posts.

## Directory Structure

- `agent.yaml`: The root configuration file defining the LoopAgent 
- `sub_agents/ml_agent.yaml`: Configuration for the central ML agent handling all logic, tooling, and execution.
- `prompts/ml_agent.md`: The system prompt for the ML agent. Contains instructions for the execution loop and exact Deotte FE and model recipes.
- `configs/`: Sampling parameters for the LLM.
- `skills/ensemble-manager/`: Python skills extending the agent's capabilities:
  - `search_policy.py`: Decides the next best task type and algorithm guide based on the history tree.
  - `hill_climb.py`: Executes the final greedy OOF ensembling.
  - `data_preview.py`: Extracts raw dataset metadata for the agent to read.
  - `update_feedback.py`: State-tracking script that persists execution logs and errors.