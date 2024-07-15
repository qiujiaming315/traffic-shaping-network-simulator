import numpy as np
import os
from pathlib import Path

"""Generate flow profiles (rate, burst, end-to-end deadline) as optimization input."""


def generate_random_flow(num_flow, seed=None):
    """
    Generate random flow profiles.
    :param num_flow: the number of flows in the generated profile.
    :param seed: the seed for random generator.
    :return: a numpy matrix describing the flow profile.
    """
    # Set the bounds for randomly generating flow profiles.
    rate_bound = (1, 100)
    burst_bound = (1, 100)
    # Configurations of deadline classes (uncomment the second line to switch from configuration 1 to 2).
    deadline_class = np.array([0.01, 0.1, 1])
    # deadline_class = np.array([0.01, 0.025, 0.05, 0.1])
    rstate = np.random.RandomState(seed)
    flow = np.zeros((num_flow, 3))
    rand_data = rstate.rand(num_flow, 2)
    # Randomly select the rate, burst, and deadline for each flow.
    flow[:, 0] = np.around(rand_data[:, 0] * (rate_bound[1] - rate_bound[0]) + rate_bound[0], 2)
    flow[:, 1] = np.around(rand_data[:, 1] * (burst_bound[1] - burst_bound[0]) + burst_bound[0], 2)
    flow[:, 2] = deadline_class[rstate.randint(len(deadline_class), size=num_flow)]
    return flow


def save_file(output_path, file_name, flow):
    """
    Save the generated flow profile to the specified output location.
    :param output_path: the directory to save the flow profile.
    :param file_name: name of the file to save the flow profile.
    :param flow: the flow profile.
    """
    Path(output_path).mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(output_path, file_name + ".npy"), flow)
    return


if __name__ == "__main__":
    num_flow = 60
    path = f"../data/flow/heterogeneous/{num_flow}/"
    # flow_data = np.array([50, 50, 0.1]) * np.ones((num_flow, 3))
    # save_file(path, "flow1", flow_data)
    for file_idx in range(10):
        save_file(path, f"flow{file_idx + 1}", generate_random_flow(num_flow))
