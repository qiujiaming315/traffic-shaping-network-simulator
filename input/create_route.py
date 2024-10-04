import numpy as np
import os
from pathlib import Path

"""Generate flow routes as optimization input."""
google_nodes = np.zeros((22,), dtype=bool)
google_nodes[11:] = True
google_links = [(0, 1), (0, 3), (0, 5), (1, 2), (1, 3), (1, 5), (2, 3), (2, 4), (3, 4), (3, 5), (3, 7), (4, 5), (4, 7),
                (4, 10), (5, 6), (5, 7), (5, 8), (6, 7), (6, 8), (6, 9), (7, 8), (7, 10), (8, 9), (0, 11), (1, 12),
                (2, 13), (3, 14), (4, 15), (5, 16), (6, 17), (7, 18), (8, 19), (9, 20), (10, 21)]
cev_nodes = np.zeros((44,), dtype=bool)
cev_nodes[13:] = True
cev_links = [(0, 4), (0, 5), (1, 4), (1, 5), (2, 4), (2, 5), (3, 4), (3, 5), (4, 6), (5, 7), (6, 8), (6, 11), (7, 9),
             (7, 11), (8, 10), (9, 12), (0, 13), (0, 14), (1, 15), (1, 16), (1, 17), (1, 18), (1, 19), (2, 20), (2, 21),
             (2, 22), (3, 23), (3, 24), (4, 25), (5, 26), (5, 27), (6, 28), (6, 29), (6, 30), (7, 31), (7, 32), (7, 33),
             (8, 34), (8, 35), (9, 36), (9, 37), (10, 38), (10, 39), (11, 40), (11, 41), (12, 42), (12, 43)]


def shortest_path_routing(nodes, links):
    """
    Compute the routing table based on shortest path routing.
    :param nodes: the network nodes.
    :param links: the network links.
    :return: The shortest route between each source and destination pair.
    """
    # Create the adjacency matrix according to the specified topology.
    adjacency_matrix = np.zeros((len(nodes), len(nodes)), dtype=bool)
    for (node1, node2) in links:
        adjacency_matrix[node1, node2] = True
        adjacency_matrix[node2, node1] = True
    # Create the routing table using the shortest paths (minimum hop routing).
    routing_table = np.where(adjacency_matrix, np.arange(len(nodes)), -1)
    hop_count = adjacency_matrix.astype(int)
    for node_idx, mask in enumerate(adjacency_matrix):
        mask = mask.copy()
        # Extract the reachable nodes and randomly shuffle them.
        reachable_nodes = np.arange(len(nodes))[mask]
        np.random.shuffle(reachable_nodes)
        queue = list(reachable_nodes)
        mask[node_idx] = True
        while len(queue):
            node = queue.pop(0)
            # Extract the reachable new nodes and randomly shuffle them.
            new_nodes = np.arange(len(nodes))[adjacency_matrix[node]]
            np.random.shuffle(new_nodes)
            for n in new_nodes:
                if not mask[n]:
                    mask[n] = True
                    routing_table[node_idx, n] = routing_table[node_idx, node]
                    hop_count[node_idx, n] = hop_count[node_idx, node] + 1
                    queue.append(n)
    # Retrieve the route between each source and destination (S-D) pair.
    sd_routes = dict()
    for src_idx in range(len(nodes)):
        for dest_idx in range(len(nodes)):
            if src_idx != dest_idx:
                route = list()
                next_node, table = src_idx, routing_table[:, dest_idx]
                while table[next_node] != -1:
                    route.append(next_node)
                    next_node = table[next_node]
                route.append(next_node)
                sd_routes[(src_idx, dest_idx)] = route
    return sd_routes


def generate_tandem_route(num_flow, num_hop, end_host=False):
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
    if end_host:
        # Append end devices to each hop.
        num_flow, num_link = flow_routes.shape
        end_links = np.zeros((num_flow, 0), dtype=int)
        hop_traversed = np.amax(flow_routes, axis=1)[:, np.newaxis]
        flow_joined, flow_departed = np.zeros((num_flow, 1), dtype=int), np.zeros((num_flow, 1), dtype=int)
        for link_idx in range(num_link):
            link_flows = flow_routes[:, link_idx:link_idx + 1]
            flow_joined = np.where(link_flows == 1, 1, 0)
            end_links = np.concatenate((end_links, flow_joined, flow_departed), axis=1)
            flow_departed = np.where(link_flows == hop_traversed, hop_traversed + 2, 0)
        end_links = np.concatenate((end_links, np.zeros((num_flow, 1), dtype=int), flow_departed), axis=1)
        flow_routes[flow_routes > 0] += 1
        flow_routes = np.concatenate((flow_routes, end_links), axis=1)
    return flow_routes


def generate_dc_net(net_nodes, net_links, num_pair, source_edge=True, dest_edge=True, routing="shortest_path",
                    routing_path="", prune=True):
    """
    Generate flow routes on the inter-datacenter network topology.
    Routes are computed through shortest path.
    :param net_nodes: a boolean array that indicates which nodes are edge nodes.
    :param net_links: a list of links in the network.
    :param num_pair: the number of S-D pairs (and corresponding flow routes) to generate.
    :param source_edge: whether the source node can only be selected from edge nodes.
    :param dest_edge: whether the destination node can only be selected from edge nodes.
    :param routing: the routing protocol. Available choices include "shortest_path" and "custom".
    :param routing_path: path to load the custom routes between S-D pairs. Active only when routing="custom".
    :param prune: Whether to remove the end-hosts from the routes.
    :return: a numpy matrix describing the routes of the flows.
    """
    # Retrieve the routes based on routing protocol.
    if routing == "custom" and os.path.isfile(routing_path):
        route_data = np.load(routing_path)
        routes = dict()
        for route_key in route_data.files:
            route = route_data[route_key]
            routes[(route[0], route[-1])] = route
    else:
        routes = shortest_path_routing(net_nodes, net_links)
    # Create a link map.
    link_map = dict()
    for link_idx, link in enumerate(net_links):
        link_map[link] = 2 * link_idx
        link_map[(link[1], link[0])] = 2 * link_idx + 1
    # Create the flow routes.
    flow_routes = np.zeros((0, 2 * len(net_links)), dtype=int)
    sd_route = np.zeros((1, 2 * len(net_links)), dtype=int)
    flow_routes_pruned = np.zeros((0, 2 * (len(net_links) - np.sum(net_nodes))), dtype=int)
    sd_route_pruned = np.zeros((1, 2 * (len(net_links) - np.sum(net_nodes))), dtype=int)
    # min_hop = 4 if prune else 2
    min_hop = 3 if prune else 1
    # Select multiple (num_pair) S-D pairs for the network.
    for _ in range(num_pair):
        sd_route[:], sd_route_pruned[:] = 0, 0
        # Randomly select a source node.
        source = np.random.choice(np.arange(len(net_nodes))[net_nodes]) if source_edge else np.random.randint(
            len(net_nodes))
        while True:
            # Select a random destination with the path between the corresponding S-D pair covering at least 2 hops.
            destination = np.random.choice(np.arange(len(net_nodes))[net_nodes]) if dest_edge else np.random.randint(
                len(net_nodes))
            if source != destination and len(routes[(source, destination)]) > min_hop:
                break
        # Establish the path according to the route.
        route = routes[(source, destination)]
        for link_count, (node, next_node) in enumerate(zip(route[:-1], route[1:])):
            sd_route[0, link_map[(int(node), int(next_node))]] = link_count + 1
        # Select a random number of flows for each S-D pair.
        # num_flow = np.random.randint(3, 11)
        # net = np.concatenate((net, np.repeat(sd_route, num_flow, axis=0)), axis=0)
        flow_routes = np.concatenate((flow_routes, sd_route), axis=0)
        if prune:
            route_pruned = route[1:-1]
            for link_count, (node, next_node) in enumerate(zip(route_pruned[:-1], route_pruned[1:])):
                sd_route_pruned[0, link_map[(int(node), int(next_node))]] = link_count + 1
            flow_routes_pruned = np.concatenate((flow_routes_pruned, sd_route_pruned), axis=0)
    flow_routes, flow_routes_pruned = flow_routes.astype(int), flow_routes_pruned.astype(int)
    return_data = {"routes": flow_routes, "routes_pruned": flow_routes_pruned} if prune else flow_routes
    return return_data


def generate_tsn_net(net_nodes, net_links, num_app, source_edge=True, dest_edge=True, routing="shortest_path",
                     routing_path="", prune=True):
    """
    Generate flow routes on the TSN setting.
    Routes are computed through shortest path.
    :param net_nodes: a boolean array that indicates which nodes are end devices.
    :param net_links: a list of links in the network.
    :param num_app: the number of messages to send through unicast, multicast, or broadcast.
    :param source_edge: whether the source node can only be selected from end devices.
    :param dest_edge: whether the destination node can only be selected from end devices.
    :param routing: the routing protocol. Available choices include "shortest_path" and "custom".
    :param routing_path: path to load the custom routes between S-D pairs. Active only when routing="custom".
    :param prune: Whether to remove the end-hosts from the routes.
    :return: a numpy matrix describing the routes of the flows.
    """
    # Retrieve the routes based on routing protocol.
    if routing == "custom" and os.path.isfile(routing_path):
        route_data = np.load(routing_path)
        routes = dict()
        for route_key in route_data.files:
            route = route_data[route_key]
            routes[(route[0], route[-1])] = route
    else:
        routes = shortest_path_routing(net_nodes, net_links)
    # Create a link map.
    link_map = dict()
    for link_idx, link in enumerate(net_links):
        link_map[link] = 2 * link_idx
        link_map[(link[1], link[0])] = 2 * link_idx + 1
    # Create the flow routes.
    flow_routes = np.zeros((0, 2 * len(net_links)), dtype=int)
    flow_routes_pruned = np.zeros((0, 2 * (len(net_links) - np.sum(net_nodes))), dtype=int)
    # min_hop = 4 if prune else 2
    min_hop = 3 if prune else 1
    # Select multiple (num_app) applications for the network.
    cast_pattern = np.random.choice(3, size=num_app)
    app_dest_num = list()
    for cast in cast_pattern:
        # Randomly select a source node.
        source = np.random.choice(np.arange(len(net_nodes))[net_nodes]) if source_edge else np.random.randint(
            len(net_nodes))
        # Randomly select destination node(s) according to unicast, multicast, or broadcast.
        dest_mask = net_nodes.copy() if dest_edge else np.ones_like(net_nodes)
        for dest_idx in range(len(net_nodes)):
            if not net_nodes[dest_idx] or source == dest_idx or len(routes[(source, dest_idx)]) <= min_hop:
                dest_mask[dest_idx] = False
        dest_candidate = np.arange(len(net_nodes))[dest_mask]
        if cast == 0 or len(dest_candidate) == 1:  # Unicast.
            dest_num = 1
            destination = np.random.choice(dest_candidate, size=1)
        elif cast == 1 and len(dest_candidate) > 2:  # Multicast.
            dest_num = np.random.randint(len(dest_candidate) - 2) + 2
            destination = np.random.choice(dest_candidate, size=dest_num, replace=False)
        else:  # Broadcast.
            dest_num = len(dest_candidate)
            destination = dest_candidate
        app_dest_num.append(dest_num)
        # Establish the path according to the route.
        sd_route = np.zeros((dest_num, 2 * len(net_links)), dtype=int)
        for dest_idx, dest in enumerate(destination):
            route = routes[(source, dest)]
            for link_count, (node, next_node) in enumerate(zip(route[:-1], route[1:])):
                sd_route[dest_idx, link_map[(int(node), int(next_node))]] = link_count + 1
        flow_routes = np.concatenate((flow_routes, sd_route), axis=0)
        if prune:
            sd_route_pruned = np.zeros((dest_num, 2 * (len(net_links) - np.sum(net_nodes))), dtype=int)
            for dest_idx, dest in enumerate(destination):
                route_pruned = routes[(source, dest)][1:-1]
                for link_count, (node, next_node) in enumerate(zip(route_pruned[:-1], route_pruned[1:])):
                    sd_route_pruned[dest_idx, link_map[(int(node), int(next_node))]] = link_count + 1
            flow_routes_pruned = np.concatenate((flow_routes_pruned, sd_route_pruned), axis=0)
    flow_routes, flow_routes_pruned = flow_routes.astype(int), flow_routes_pruned.astype(int)
    return_data = {"routes": flow_routes, "app_dest_num": app_dest_num}
    if prune:
        return_data["routes_pruned"] = flow_routes_pruned
    return return_data


def generate_google_net(num_pair, routing="shortest_path", routing_path="", prune=False):
    """
    Generate flow routes using the Google (US) network topology.
    The google network topology is motivated by https://cloud.google.com/about/locations#network.
    """
    return generate_dc_net(google_nodes, google_links, num_pair, source_edge=True, dest_edge=True, routing=routing,
                           routing_path=routing_path, prune=prune)


def generate_cev_net(num_pair, routing="shortest_path", routing_path="", prune=False):
    """
    Generate flow routes using the orion CEV network topology.
    The paper is available at https://ieeexplore.ieee.org/abstract/document/8700610/.
    """
    return generate_tsn_net(cev_nodes, cev_links, num_pair, source_edge=True, dest_edge=True, routing=routing,
                            routing_path=routing_path, prune=prune)


def save_file(output_path, file_name, flow_routes):
    """
    Save the generated flow routes to the specified output location.
    :param output_path: the directory to save the flow routes.
    :param file_name: name of the file to save the flow routes.
    :param flow_routes: the flow routes.
    """
    Path(output_path).mkdir(parents=True, exist_ok=True)
    if isinstance(flow_routes, dict):
        np.savez(os.path.join(output_path, file_name + ".npz"), **flow_routes)
    else:
        np.save(os.path.join(output_path, file_name + ".npy"), flow_routes)
    return


if __name__ == "__main__":
    # First, specify the directory to save the generated flow routes.
    path = "./route/"
    file_name = "route1"
    # Specify the seed for the random number generator.
    np.random.seed(0)
    # You can specify your own flow routes and directly save it to the directory.
    route = np.array([[2, 1, 0],
                      [1, 2, 3],
                      [0, 1, 0]
                      ])
    save_file(path, file_name, route)
    # Alternatively, you may generate flow routes in a tandem network.
    save_file(path, file_name, generate_tandem_route(5, 3))
    # Or you can generate flow routes motivated by some realistic network topology.
    save_file(path, file_name, generate_google_net(10))  # For the US-Topo (inter-datacenter).
    save_file(path, file_name, generate_cev_net(10))  # For the Orion CEV network (TSN setting for Ethernet).
