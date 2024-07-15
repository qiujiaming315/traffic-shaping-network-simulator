import numpy as np
import os
from pathlib import Path

"""Generate flow routes as optimization input."""


def generate_tandem_route(num_flow, num_hop):
    """
    Generate flow routes on the tandem network topology (i.e., parking-lot network).
    """
    # Create the main flows in the network.
    net_main = np.ones((2 * num_flow, 1), dtype=int) * (np.arange(num_hop) + 1)
    # Create cross flows in the network.
    net_cross = np.zeros((num_hop + 1, num_hop), dtype=int)
    net_cross[np.arange(num_hop - 1) + 1, np.arange(num_hop - 1)] = 1
    net_cross[np.arange(num_hop - 1) + 1, np.arange(num_hop - 1) + 1] = 2
    net_cross[0, 0] = 1
    net_cross[-1, -1] = 1
    # Create multiple cross flows at each entry.
    net_cross = np.repeat(net_cross, num_flow, axis=0)
    # Combine the main and cross flows.
    flow_routes = np.concatenate((net_main, net_cross), axis=0)
    return flow_routes


def save_file(output_path, file_name, flow_routes):
    """
    Save the generated flow routes to the specified output location.
    :param output_path: the directory to save the flow routes.
    :param file_name: name of the file to save the flow routes.
    :param flow_routes: the flow routes.
    """
    Path(output_path).mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(output_path, file_name + ".npy"), flow_routes)
    return


if __name__ == "__main__":
    num_cross_flow = 5
    num_hop = 9
    route = generate_tandem_route(num_cross_flow, num_hop)
    # The shape of the route matrix should be ((num_hop + 3) * num_flow, num_hop)
    num_flow, num_hop = route.shape
    path = f"../data/route/{num_flow}/"
    save_file(path, "route1", route)
