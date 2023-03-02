import re


class CVRP:
    def __init__(
        self,
        name,
        comment,
        type,
        dimension,
        edge_weight_type,
        capacity,
        node_coord_section,
        demand_section,
        depot_section,
        vehicles,
    ):
        self.name = name
        self.comment = comment
        self.type = type
        self.dimension = dimension
        self.edge_weight_type = edge_weight_type
        self.capacity = capacity
        self.node_coord_section = node_coord_section
        self.demand_section = demand_section
        self.depot_section = depot_section
        self.vehicles = vehicles

    def calculate_distance(self, opts=None, tsplib_distances_type="EUC_2D"):
        self.node_coord_section = np.asarray(self.node_coord_section)
        pdtype = "euclidean"
        postprocess = lambda M: M  # noqa E731

        if tsplib_distances_type == "MAX_2D":
            pdtype = "chebyshev"
        elif tsplib_distances_type == "MAN_2D":
            pdtype = "cityblock"
        elif tsplib_distances_type == "CEIL_2D":
            postprocess = lambda D: np.ceil(D).astype(int)  # noqa E731
        elif tsplib_distances_type == "FLOOR_2D":
            postprocess = lambda D: np.floor(D).astype(int)  # noqa E731
        elif tsplib_distances_type == "EUC_2D":
            postprocess = lambda D: np.round(D).astype(int)  # noqa E731
        elif tsplib_distances_type == "EXACT_2D":
            pass
        else:
            raise ValueError("Unknown distance method")
        if opts is None:
            return postprocess(squareform(pdist(self.node_coord_section, pdtype)))
        else:
            return postprocess(cdist(self.node_coord_section, opts, pdtype))


def read_cvrp(file_name):
    """
    Args:

       file_name: CVRP instance filename

    Returns:
    """
    problem_details = {
        "name": "",
        "comment": "",
        "type": "",
        "dimension": "",
        "edge_weight_type": "",
        "capacity": "",
        "demand_section": [],
        "node_coord_section": [],
        "depot_section": "",
    }

    node_coord_section = False
    demand_section = False
    depot_section = False

    r = re.search("(?<=k)[0-9]+(?=\.vrp)", file_name)
    problem_details["vehicles"] = int(r.group(0))
    splitting_char = " "
    with open(file_name, "r") as f:
        while line := f.readline().strip():
            if ":" in line:
                key, value = line.split(":", 1)
                problem_details[key.strip().lower()] = value.strip()

            elif line == "NODE_COORD_SECTION":
                node_coord_section = True
                demand_section = False
                depot_section = False
                continue
            elif line == "DEMAND_SECTION":
                demand_section = True
                node_coord_section = False
                depot_section = False
                continue
            elif line == "DEPOT_SECTION":
                depot_section = True
                demand_section = False
                node_coord_section = False
                continue

            if node_coord_section:
                line = [
                    value for value in line.split(splitting_char) if value.strip() != ""
                ]
                _, x_coord, y_coord = line
                problem_details["node_coord_section"].append(
                    [float(x_coord), float(y_coord)]
                )

            if demand_section:
                demand_section = True
                node, demand = line.split(splitting_char)
                problem_details["demand_section"].append(int(demand))

            if depot_section:
                problem_details["depot_section"] = line

        cvrp_instance = CVRP(**problem_details)
        return cvrp_instance
