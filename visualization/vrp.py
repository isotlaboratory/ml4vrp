import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from numpy.random import shuffle


class VRPVisualizer:
    def __init__(self, num_nodes) -> None:
        self.num_nodes = num_nodes
        fig = plt.figure(constrained_layout=True)
        fig.set_size_inches(20, 12)
        self.ax = fig.subplot_mosaic(
            [
                ["gnn"],
            ],
        )

        self.colors = list(mcolors.CSS4_COLORS.keys())[:num_nodes]
        shuffle(self.colors)

    def __call__(
        self, scip_model, gap, dual_bound, primal_bound, coords, xfeas, xpresolve, paths
    ):
        plt.title(
            f"Gap: {gap if not scip_model.isInfinity(gap) else 'Infinity'}, Dual Bound: {dual_bound if not scip_model.isInfinity(dual_bound) else 'Infinity'}, Primal Bound Gap: {primal_bound if not scip_model.isInfinity(primal_bound) else 'Infinity'}"
        )
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    if not scip_model.isInfinity(gap):
                        if xfeas[i][j] == 1.0:
                            xc, yc = zip(*[coords[i], coords[j]])
                            if i == 0 or j == 0:
                                self.ax["gnn"].plot(
                                    xc,
                                    yc,
                                    f"gray",
                                    label="Feasible Solution",
                                    linewidth=2,
                                )
                            else:
                                path_idx = None
                                for idx, path in enumerate(paths):
                                    if i in path and j in path:
                                        path_idx = idx
                                color = self.colors[path_idx]
                                self.ax["gnn"].plot(
                                    xc,
                                    yc,
                                    f"{color}",
                                    label="Feasible Solution",
                                )
                        if xpresolve[i][j]:
                            xc, yc = zip(*[coords[i], coords[j]])
                            self.ax["gnn"].plot(xc, yc, "bo-", alpha=0.1)
                    else:
                        if xfeas[i][j] == 1.0:
                            xc, yc = zip(*[coords[i], coords[j]])
                            self.ax["gnn"].plot(xc, yc, f"mo-", "Integer Solution")
                        if xpresolve[i][j]:
                            xc, yc = zip(*[coords[i], coords[j]])
                            self.ax["gnn"].plot(xc, yc, "bo-", alpha=0.1)
        plt.pause(0.0001)
        self.ax["gnn"].lines.clear()
