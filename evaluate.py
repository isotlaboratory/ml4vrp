import argparse
import csv
import os
import re
import time

import ecole
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything

import modules
from utils import read_cvrp
from visualization.bpp import BPPVisualizer
from visualization.vrp import VRPVisualizer

seed = 2
seed_everything(seed)

parser = argparse.ArgumentParser(description="AI4L Dataset Generation")

parser.add_argument(
    "--checkpoint",
    dest="checkpoint",
    type=str,
    default=None,
    help="path to checkpoint",
    required=True,
)

parser.add_argument(
    "--arch",
    dest="arch",
    type=str,
    choices=["GCNN", "GraphSAGE", "GAT"],
    help="Geometric Deep Learning model choice",
    required=True,
)

parser.add_argument(
    "--results-path",
    dest="results_path",
    type=str,
    default=".",
    help="Path to pre sampled instances",
)

parser.add_argument(
    "--time-limit",
    dest="time_limit",
    type=int,
    help="Time limit for a single sampling episode",
)

parser.add_argument(
    "--dataset",
    dest="dataset",
    type=str,
    help="Path to the dataset",
    required=True,
)

parser.add_argument(
    "--problem",
    dest="problem",
    choices=["VRP", "BPP"],
    help="Optimization problem",
    required=True,
)

parser.add_argument(
    "--live",
    dest="live",
    action="store_true",
    help="Visualization of the solution process while evaluating the model",
)

args = parser.parse_args()

instance_path = args.dataset
dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]

if args.problem == "VRP":
    cvrp = read_cvrp(instance_path)
    coords = cvrp.node_coord_section
    num_nodes = int(cvrp.dimension)
    binvars_count = num_nodes**2 - num_nodes
    indices = np.diag_indices(num_nodes)
    # feasible solution
    xfeas = np.zeros((num_nodes, num_nodes))
    # relaxed problem solution
    xpresolve = np.zeros((num_nodes, num_nodes))
    mask = np.ones(xfeas.shape, bool)
    mask[indices] = False

    r = re.search("(?<=k)[0-9]+(?=\.vrp)", args.dataset)
    num_vehicles = int(r.group(0))
    instances = ecole.instance.CapacitatedVehicleRoutingLoader(
        instance_path, num_vehicles
    )
    visualizer = VRPVisualizer(num_nodes=num_nodes)

elif args.problem == "BPP":
    # relaxed problem solution
    f = open(args.dataset)
    lines = f.readlines()
    capacity = int(float(lines[1].strip().split(" ")[0]))
    num_items = int(re.match(r".*[ut]([0-9]+).*", args.dataset).group(1))
    items_weight = np.array(list(map(float, lines[2:])))
    # reading the number of bins from the second line third column after stripping the LF char.
    num_bins = int(lines[1].strip().split(" ")[2])
    f.close()

    binvars_count = num_bins * num_items
    # feasible solution
    xfeas = np.zeros((num_bins, num_items))
    instances = ecole.instance.Binpacking(instance_path, num_bins)
    visualizer = BPPVisualizer(num_bins, num_items, capacity, items_weight)
else:
    raise ValueError("Unrecognized Problem")

results_path = args.results_path
os.makedirs(results_path, exist_ok=True)

time_limit = args.time_limit
arch = args.arch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = (
    modules.__dict__[f"MIPPLModel{arch}"]
    .load_from_checkpoint(args.checkpoint)
    .to(device)
)

instance = next(instances)
PRESOLVE_FEATURE_IDX = 8
result_file = f"{dataset_name}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
fieldnames = [
    "type",
    "instance",
    "nnodes",
    "treesize",
    "nlps",
    "stime",
    "gap",
    "status",
    "walltime",
    "proctime",
    "primal_bound",
    "dual_bound",
    "checkpoint",
    "time_limit",
]
os.makedirs(f"{results_path}/results", exist_ok=True)
scip_parameters = {
    "estimation/treeprofile/enabled": True,
    "separating/maxrounds": 0,
    "presolving/maxrestarts": 0,
    "limits/time": time_limit,
    "timing/clocktype": 1,
    "branching/vanillafullstrong/idempotent": True,
}

env = ecole.environment.Branching(
    observation_function=ecole.observation.NodeBipartite(),
    information_function={
        "nb_nodes": ecole.reward.NNodes(),
        "time": ecole.reward.SolvingTime(),
        "tree_size_estimate": ecole.reward.TreeSizeEstimate(),
    },
    scip_params=scip_parameters,
)

env.seed(seed)

with open(f"{results_path}/results/{result_file}", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    # Run the learned brancher
    nb_nodes = 0
    tree_size_estimate = 0
    walltime = time.perf_counter()
    proctime = time.process_time()
    observation, action_set, _, done, info = env.reset(instance)

    nb_nodes += info["nb_nodes"]
    while not done:
        if args.live:
            scip_model = env.model.as_pyscipopt()
            stime = scip_model.getSolvingTime()
            nnodes = scip_model.getNNodes()
            nlps = scip_model.getNLPs()
            gap = scip_model.getGap()
            status = scip_model.getStatus()
            primal_bound = scip_model.getPrimalbound()
            dual_bound = scip_model.getDualbound()
            best_sol = scip_model.getBestSol()

            if args.problem == "VRP":
                contvars = observation.variable_features[binvars_count:]
                binvars_scip = scip_model.getVars()[:binvars_count]
                binvars_ecole = observation.variable_features[:binvars_count]
                sol_val = list(map(lambda var: best_sol[var], binvars_scip))

                xfeas[mask] = np.array(sol_val)
                # the eighth feature is the relaxed problem solution value
                xpresolve[mask] = binvars_ecole[:, PRESOLVE_FEATURE_IDX]

                paths = []
                if not scip_model.isInfinity(gap):
                    g = nx.from_numpy_array(xfeas)
                    paths = nx.cycle_basis(g)

                visualizer(
                    scip_model,
                    gap,
                    dual_bound,
                    primal_bound,
                    coords,
                    xfeas,
                    xpresolve,
                    paths,
                )
            if args.problem == "BPP":
                binvars_scip = scip_model.getVars()[num_bins:]
                sol_val = list(map(lambda var: best_sol[var], binvars_scip))
                xfeas = np.array(sol_val).reshape(num_bins, num_items)
                bins, _ = (xfeas == 1).nonzero()
                bins = np.unique(bins)
                xpresolve = observation.variable_features[
                    num_bins:, PRESOLVE_FEATURE_IDX
                ].reshape(num_bins, num_items)
                visualizer(scip_model, gap, bins, xfeas, xpresolve)

        with torch.no_grad():
            observation = (
                torch.from_numpy(observation.row_features.astype(np.float32)).to(
                    device
                ),
                torch.from_numpy(observation.edge_features.indices.astype(np.int64)).to(
                    device
                ),
                torch.from_numpy(observation.edge_features.values.astype(np.float32))
                .view(-1, 1)
                .to(device),
                torch.from_numpy(observation.variable_features.astype(np.float32)).to(
                    device
                ),
            )
            # GCNN natively accepts bipartite graphs
            if arch != "GCNN":
                constraint_features = F.pad(
                    observation[0], (1, 13), "constant", 0
                )  # pad with zeros the amount of features that makes variable features and constraint features equal in size.
                features = torch.vstack([observation[3], constraint_features])
                adj = policy._pad_adj(
                    observation[1],  # edge index
                    observation[2],  # edge features
                    observation[3].size(0),  # variable features
                    observation[0].size(0),  # constraint features
                )
                logits = policy(features, adj.indices(), adj.values())
            else:
                logits = policy(*observation)

            action = action_set[logits[action_set.astype(np.int64)].argmax()]
            observation, action_set, _, done, info = env.step(action)
        tree_size_estimate = info["tree_size_estimate"]
        nb_nodes += info["nb_nodes"]

    walltime = time.perf_counter() - walltime
    proctime = time.process_time() - proctime
    scip_model = env.model.as_pyscipopt()
    stime = scip_model.getSolvingTime()
    nnodes = scip_model.getNNodes()
    nlps = scip_model.getNLPs()
    gap = scip_model.getGap()
    status = scip_model.getStatus()
    primal_bound = scip_model.getPrimalbound()
    dual_bound = scip_model.getDualbound()
    writer.writerow(
        {
            "type": "gnn",
            "instance": instance_path,
            "nnodes": nnodes,
            "treesize": tree_size_estimate,
            "nlps": nlps,
            "stime": stime,
            "gap": gap,
            "status": status,
            "walltime": walltime,
            "proctime": proctime,
            "primal_bound": primal_bound,
            "dual_bound": dual_bound,
            "checkpoint": args.checkpoint,
            "time_limit": time_limit,
        }
    )
    csvfile.flush()
