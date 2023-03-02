import threading

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


class BPPVisualizer:
    def __init__(self, num_bins, num_items, capacity, items_weight) -> None:
        self.rows = 6
        self.cols = 8
        self.num_bins = num_bins
        self.capacity = capacity
        self.items_weight = items_weight
        self.fig = plt.figure(constrained_layout=True)
        self.fig.set_size_inches(20, 12)
        self.ax = self.fig.subplots(
            self.rows, self.cols, squeeze=False, subplot_kw={"projection": "3d"}
        )

        edges, vertices = self._edges_only_cube()
        for row in range(self.rows):
            for col in range(self.cols):
                bin_number = row * self.cols + col
                self.ax[row, col].set_title(f"Bin {bin_number}")
                self.ax[row, col].set_xlim(0, self.capacity)
                self.ax[row, col].set_ylim(0, self.capacity)
                self.ax[row, col].set_zlim(0, self.capacity)
                # plot the faces of the cube
                for edge in edges:
                    self.ax[row, col].plot(
                        *zip(vertices[edge[0]], vertices[edge[1]]), color="k"
                    )
        color_list = list(mcolors.CSS4_COLORS.keys())[:num_items]
        # Create a dictionary with the index of the color as the key
        self.colors = {i: color for i, color in enumerate(color_list)}

    def _edges_only_cube(self):
        # Define the vertices of the cube
        vertices = [
            (0, 0, 0),
            (0, 0, self.capacity),
            (0, self.capacity, 0),
            (0, self.capacity, self.capacity),
            (self.capacity, 0, 0),
            (self.capacity, 0, self.capacity),
            (self.capacity, self.capacity, 0),
            (self.capacity, self.capacity, self.capacity),
        ]

        # Define the edges of the cube
        edges = [
            (0, 1),
            (0, 2),
            (0, 4),
            (1, 3),
            (1, 5),
            (2, 3),
            (2, 6),
            (3, 7),
            (4, 5),
            (4, 6),
            (5, 7),
            (6, 7),
        ]

        # Draw the vertices and edges of the cube

        return edges, vertices

    def _plot_items(self, xfeas, xpresolve, bin_num):
        # non zero returns a tuple
        bin_items = xfeas[bin_num].nonzero()[0]
        # just converting the bin number back to how the mosaic plot was constructed
        bin_row = bin_num // self.cols
        bin_col = bin_num % self.cols
        # getting each item unique color to track movements across bins
        items_colors = [self.colors[index] for index in bin_items]
        # sorting items by weight so that they'd look good in the plot
        sorting_indices = np.argsort(self.items_weight[bin_items])
        # descending order of item weight
        bin_items_weights = self.items_weight[bin_items][sorting_indices][::-1]
        # sorting item colors by the same indices
        items_colors = [items_colors[i] for i in sorting_indices]
        bin_items_weights_color = zip(bin_items_weights, items_colors)
        # feasible solution plot
        for item_index, (item_weight, color) in enumerate(bin_items_weights_color):
            # 0 if first item, or previous item weight as a z_shift
            z_shift = 0 if item_index == 0 else bin_items_weights[:item_index].sum()
            start = 0
            end = item_weight
            points = np.array(
                [
                    # X     Y       Z
                    # bottom face points
                    [start, start, start + z_shift],
                    [end, start, start + z_shift],
                    [end, end, start + z_shift],
                    [start, end, start + z_shift],
                    # top face points
                    [start, start, end + z_shift],
                    [end, start, end + z_shift],
                    [end, end, end + z_shift],
                    [start, end, end + z_shift],
                ]
            )

            r = [start, end]
            X, Y = np.meshgrid(r, r)
            one = np.ones(4).reshape(2, 2) * end
            some_number = np.ones(4).reshape(2, 2) * start
            # top face
            self.ax[bin_row, bin_col].plot_surface(
                X, Y, one + z_shift, alpha=1.0, color=color
            )
            # bottom face
            self.ax[bin_row, bin_col].plot_surface(
                X, Y, some_number + z_shift, alpha=1.0, color=color
            )
            # front face
            self.ax[bin_row, bin_col].plot_surface(
                X, some_number, Y + z_shift, alpha=1.0, color=color
            )
            # back face
            self.ax[bin_row, bin_col].plot_surface(
                X, one, Y + z_shift, alpha=1.0, color=color
            )
            # right face
            self.ax[bin_row, bin_col].plot_surface(
                one, X, Y + z_shift, alpha=1.0, color=color
            )
            # left face
            self.ax[bin_row, bin_col].plot_surface(
                some_number, X, Y + z_shift, alpha=1.0, color=color
            )
            self.ax[bin_row, bin_col].scatter3D(
                points[:, 0], points[:, 1], points[:, 2], color=color, s=5
            )
        # presolve solution plot
        presolve_color = "b"
        presolve_alpha = 0.1

        presolve_bin_items = xpresolve[bin_num].nonzero()[0]
        # sorting items by weight so that they'd look good in the plot
        sorting_indices = np.argsort(self.items_weight[presolve_bin_items])
        # descending order of item weight
        presolve_bin_items_weights = self.items_weight[presolve_bin_items][
            sorting_indices
        ][::-1]

        for presolve_item_index, presolve_item_weight in enumerate(
            presolve_bin_items_weights
        ):
            # 0 if first item, or previous item weight as a z_shift
            z_shift = (
                0
                if presolve_item_index == 0
                else presolve_bin_items_weights[:presolve_item_index].sum()
            )
            start = 0
            end = presolve_item_weight
            points = np.array(
                [
                    # X     Y       Z
                    # bottom face points
                    [self.capacity - start, self.capacity - start, start + z_shift],
                    [self.capacity - end, self.capacity - start, start + z_shift],
                    [self.capacity - end, self.capacity - end, start + z_shift],
                    [self.capacity - start, self.capacity - end, start + z_shift],
                    # top face points
                    [self.capacity - start, self.capacity - start, end + z_shift],
                    [self.capacity - end, self.capacity - start, end + z_shift],
                    [self.capacity - end, self.capacity - end, end + z_shift],
                    [self.capacity - start, self.capacity - end, end + z_shift],
                ]
            )

            r = [start, end]
            X, Y = np.meshgrid(r, r)
            one = np.ones(4).reshape(2, 2) * end
            some_number = np.ones(4).reshape(2, 2) * start
            # top face
            self.ax[bin_row, bin_col].plot_surface(
                self.capacity - X,
                self.capacity - Y,
                one + z_shift,
                alpha=presolve_alpha,
                color=presolve_color,
            )
            # bottom face
            self.ax[bin_row, bin_col].plot_surface(
                self.capacity - X,
                self.capacity - Y,
                some_number + z_shift,
                alpha=presolve_alpha,
                color=presolve_color,
            )
            # front face
            self.ax[bin_row, bin_col].plot_surface(
                self.capacity - X,
                self.capacity - some_number,
                Y + z_shift,
                alpha=presolve_alpha,
                color=presolve_color,
            )
            # back face
            self.ax[bin_row, bin_col].plot_surface(
                self.capacity - X,
                self.capacity - one,
                Y + z_shift,
                alpha=presolve_alpha,
                color=presolve_color,
            )
            # right face
            self.ax[bin_row, bin_col].plot_surface(
                self.capacity - one,
                self.capacity - X,
                Y + z_shift,
                alpha=presolve_alpha,
                color=presolve_color,
            )
            # left face
            self.ax[bin_row, bin_col].plot_surface(
                self.capacity - some_number,
                self.capacity - X,
                Y + z_shift,
                alpha=presolve_alpha,
                color=presolve_color,
            )
            self.ax[bin_row, bin_col].scatter3D(
                points[:, 0], points[:, 1], points[:, 2], color=presolve_color, s=5
            )

    def __call__(self, scip_model, gap, bins, xfeas, xpresolve):
        if not scip_model.isInfinity(gap):
            threads = []
            for bin_num in bins:
                # Create a thread with the print_value function
                thread = threading.Thread(
                    target=self._plot_items, args=(xfeas, xpresolve, bin_num)
                )
                # Add the thread to the list of threads
                threads.append(thread)
                # Start the thread
                thread.start()

            # Use another for loop to wait for all threads to finish
            for thread in threads:
                thread.join()

            self.fig.text(
                0.5,
                0.5,
                f"$K_{{\\mathrm{{min}}}}$ = {len(bins)}, Gap = {gap:.3f} ",
                ha="center",
                va="center",
                fontsize=20,
            )
            # rendering
            plt.pause(0.00001)

            for bin_num in bins:
                bin_row = bin_num // self.cols
                bin_col = bin_num % self.cols
                self.ax[bin_row, bin_col].collections.clear()
