# scuola

Playground to train/run LLM agents 

Approach 


1. Setup a interactive session with finetuning image, allocating 2 nodes

2. Run Scuola.train interactively with a streaming dataset of multiple streams

3. Implement envs in dataloader so that every worker of GPU runs an env and upload to a specific location in Volume. The envs run and generate dataset, then LLM training starts as usual.

4. Implement PPO in train.py

5. Restructure the code so it is eaiser to generalize

