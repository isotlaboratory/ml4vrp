import argparse
import csv
import gzip
import logging
import os
import pickle
import re
import time
from datetime import datetime
from pathlib import Path

import ecole
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric

parser = argparse.ArgumentParser(description="VRP and BPP Training Datasets Sampler")

parser.add_argument(
    "--instance", dest="instance", type=str, required=True, help="path to instance"
)
parser.add_argument(
    "--time-limit",
    dest="time_limit",
    type=int,
    default=86_000,
    help="Time limit for a single samples collecting episode; we advise that it be sufficiently long for completing the asked `--num-train-samples`",
)

parser.add_argument(
    "--num-train-samples",
    dest="num_train_samples",
    type=int,
    default=100_000,
    help="Number of samples to collect",
)

parser.add_argument(
    "--samples-path",
    dest="samples_path",
    type=str,
    default=".",
    help="Path for exporting the B&B decisions samples",
)
parser.add_argument(
    "--problem",
    dest="problem",
    choices=["VRP", "BPP"],
    help="evaluate model",
    required=True,
)

args = parser.parse_args()

num_train_samples = args.num_train_samples

time_limit = args.time_limit  # 12 hours

instance_path = args.instance
instance_name = os.path.splitext(os.path.basename(args.instance))[0]
if args.problem == "VRP":
    r = re.search("(?<=k)[0-9]+(?=\.vrp)", args.instance)
    num_vehicles = int(r.group(0))
    instances = ecole.instance.CapacitatedVehicleRoutingLoader(
        instance_path, num_vehicles
    )
elif args.problem == "BPP":
    f = open(args.instance)
    lines = f.readlines()
    # reading the number of bins from the second line third column after stripping the LF char.
    num_bins = int(lines[1].strip().split(" ")[2])
    f.close()
    instances = ecole.instance.Binpacking(instance_path, num_bins)
else:
    raise ValueError("Unrecognized Problem")


samples_path = args.samples_path

regex_string = r"(?<=AI4L\-)[0-9]{8}"
now = datetime.now()

# ExploreThenStrongBranch class as suggested in École.
class ExploreThenStrongBranch:
    """
    This custom observation function class will randomly return either strong branching scores (expensive expert)
    or pseudocost scores (weak expert for exploration) when called at every node.
    """

    def __init__(self, expert_probability):
        self.expert_probability = expert_probability
        self.pseudocosts_function = ecole.observation.Pseudocosts()
        self.strong_branching_function = ecole.observation.StrongBranchingScores()

    def before_reset(self, model):
        """
        This function will be called at initialization of the environment (before dynamics are reset).
        """
        self.pseudocosts_function.before_reset(model)
        self.strong_branching_function.before_reset(model)

    def extract(self, model, done):
        """
        Should we return strong branching or pseudocost scores at time node?
        """
        probabilities = [1 - self.expert_probability, self.expert_probability]
        expert_chosen = bool(np.random.choice(np.arange(2), p=probabilities))
        if expert_chosen:
            return (self.strong_branching_function.extract(model, done), True)
        else:
            return (self.pseudocosts_function.extract(model, done), False)


scip_parameters = {
    "separating/maxrounds": 0,
    "presolving/maxrestarts": 0,
    "limits/time": time_limit,
    "timing/clocktype": 2,
}

# Note how we can tuple observation functions to return complex state information
env = ecole.environment.Branching(
    observation_function=(
        ExploreThenStrongBranch(expert_probability=0.50),
        ecole.observation.NodeBipartite(),
    ),
    scip_params=scip_parameters,
)

# This will seed the environment for reproducibility
env.seed(0)
episode_counter, sample_counter = 0, 0
Path(f"{samples_path}/{instance_name}-samples/").mkdir(parents=True, exist_ok=True)

# Run episodes until we have saved enough samples
start_time = time.time()
print("Sampling ...")
while sample_counter < num_train_samples:
    episode_counter += 1
    observation, action_set, _, done, _ = env.reset(next(instances))
    while not done and sample_counter < num_train_samples:
        (scores, scores_are_expert), node_observation = observation
        action = action_set[scores[action_set].argmax()]

        # Only save samples if they are coming from the expert (strong branching)
        if scores_are_expert and (sample_counter < num_train_samples):
            sample_counter += 1
            data = [node_observation, action, action_set, scores]
            filename = (
                f"{samples_path}/samples-{instance_name}/sample_{sample_counter}.pkl"
            )

            if sample_counter % int(10 * num_train_samples / 100) == 0:
                print(f"Samples collected: {sample_counter}")
            with gzip.open(filename, "wb") as f:
                pickle.dump(data, f)

        observation, action_set, _, done, _ = env.step(action)

    print(f"Episode {episode_counter}, {sample_counter} samples collected so far")
