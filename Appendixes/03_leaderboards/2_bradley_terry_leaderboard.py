# Copyright (c) Sebastian Raschka under Apache License 2.0
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

import json
import math
import argparse
import torch
from reasoning_from_scratch.ch02 import get_device

#GETS ALWAYS THE SAME RESULTS REGARDLESS of the PAIRS' ORDER!

# Step 1: Build index mapping (lines 15-17)
# models = sorted({m for winner, loser in vote_pairs for m in (winner, loser)}) #set
# # Example: ["claude", "gpt4", "llama"]

# idx = {m: i for i, m in enumerate(models)}
# # {"claude": 0, "gpt4": 1, "llama": 2}

# Step 2: Convert votes to tensor indices (lines 19-21)

# Example:
# vote_pairs = [("gpt4", "claude"), ("gpt4", "llama"), ("claude", "llama")]
# winners = tensor([1, 1, 0])  # gpt4=1, gpt4=1, claude=0
# losers = tensor([0, 2, 2])   # claude=0, llama=2, llama=2

# Step 3: Initialize learnable parameters (lines 23-24)
# theta = torch.nn.Parameter(torch.zeros(n - 1, device=device)) 
# #n-1 because last model takes 0 as fixed rating -->  def scores(): return torch.cat([theta, torch.zeros(1, device=device)])
# optimizer = torch.optim.Adam([theta], lr=0.01, weight_decay=1e-4)

# Step 4: Optimize via gradient descent (lines 29-36)
# for epoch in range(500):
#     s = scores()  # Current rating estimates
#     delta = s[winners] - s[losers]  # Score differences
#     loss = -torch.nn.functional.logsigmoid(delta).mean() #maximizes difference between the two scores (like RLHF)
#     # ... backprop and update

# Step 5: final conversion in Elo-like form:
# The raw scores from Bradley-Terry are in "log-odds" units. The formula converts them to match Elo conventions.

def bradley_terry_torch(vote_pairs, device):

    # Collect all unique model names
    models = sorted({m for winner, loser in vote_pairs for m in (winner, loser)})
    n = len(models)
    idx = {m: i for i, m in enumerate(models)}

    # Convert to index tensors
    winners = torch.tensor([idx[winner] for winner, _ in vote_pairs], dtype=torch.long)
    losers = torch.tensor([idx[loser] for _, loser in vote_pairs], dtype=torch.long)

    # Learnable parameters
    theta = torch.nn.Parameter(torch.zeros(n - 1, device=device))
    optimizer = torch.optim.Adam([theta], lr=0.01, weight_decay=1e-4)

    def scores():
        return torch.cat([theta, torch.zeros(1, device=device)])

    for epoch in range(500):
        s = scores()
        delta = s[winners] - s[losers]       # score difference
        loss = -torch.nn.functional.logsigmoid(delta).mean()   # negative log-likelihood
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Convert latent scores to Elo-like scale
    with torch.no_grad():
        s = scores()
        scale = 400.0 / math.log(10.0)
        R = s * scale
        R -= R.mean()
        R += 1000.0  # center around 1000

    return {m: float(r) for m, r in zip(models, R.cpu().tolist())}


def main():
    parser = argparse.ArgumentParser(description="Bradley-Terry leaderboard.")
    parser.add_argument("--path", type=str, help="Path to votes JSON")
    args = parser.parse_args()

    with open(args.path, "r", encoding="utf-8") as f:
        votes = json.load(f)

    device = get_device()
    ratings = bradley_terry_torch(votes, device)

    leaderboard = sorted(ratings.items(),
                         key=lambda x: -x[1])
    print("\nLeaderboard (Bradley-Terry)")
    print("-----------------------------")
    for i, (model, score) in enumerate(leaderboard, 1):
        print(f"{i:>2}. {model:<10} {score:7.1f}")
    print()


if __name__ == "__main__":
    main()
