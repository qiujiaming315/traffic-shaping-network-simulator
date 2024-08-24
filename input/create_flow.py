import numpy as np
import os
from pathlib import Path

"""Generate flow profiles (rate, burst, end-to-end deadline) as optimization input."""
cdf_values = np.arange(0, 1.01, 0.05)
fb_web_size = np.array(
    [[0.07, 0.15, 0.15, 0.2, 0.3, 0.3, 0.3, 0.5, 0.6, 0.75, 0.85, 1.2, 2, 2.9, 3.8, 7, 15, 23, 40, 90, 200]])
fb_cache_size = np.array(
    [[0.07, 0.1, 0.4, 0.8, 2, 2.2, 2.5, 2.7, 3, 3.5, 3.8, 4, 4.5, 4.8, 5, 5.8, 7, 80, 1000, 1500, 3000]])
fb_hadoop_size = np.array(
    [[0.08, 0.15, 0.22, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.4, 0.7, 1.5, 2.7, 10, 300]])
fb_web_duration = np.array(
    [[1, 1, 1, 2, 4, 40, 90, 200, 600, 2000, 10500, 20000, 40000, 60000, 80000, 100000, 130000, 140000, 150000, 150000,
      150000]])
fb_cache_duration = np.array(
    [[1, 1, 1, 100, 2700, 40000, 70000, 80000, 85000, 90000, 100000, 105000, 110000, 120000, 130000, 200000, 500000,
      550000, 600000, 600000, 600000]])
fb_hadoop_duration = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 5, 300, 3500, 15000, 35000, 100000, 600000]])
fb_size = np.concatenate((fb_web_size, fb_cache_size, fb_hadoop_size), axis=0)
fb_size = fb_size / 1000
fb_duration = np.concatenate((fb_web_duration, fb_cache_duration, fb_hadoop_duration), axis=0)
fb_duration = fb_duration / 1000
fb_deadline = np.array([0.01, 0.05, 0.2])
fb_ratio = np.array([3, 9, 1])
fb_burst = np.array([0.15, 0.4, 0.3]) / 1000
fb_burst_scale = np.array([5, 10, 1])
fb_burst = fb_burst * fb_burst_scale
fb_ratio = fb_ratio / np.sum(fb_ratio)

tsn_cdt_interval = np.array([0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
tsn_a_interval = np.array([0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
tsn_b_interval = np.array([0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
tsn_interval = [tsn_cdt_interval, tsn_a_interval, tsn_b_interval]
tsn_interval = [class_interval / 1000 for class_interval in tsn_interval]
tsn_frame_size = np.array([128, 256, 256]) / 1e6
tsn_deadline = np.array([0.1, 2, 50]) / 1000
tsn_ratio = np.array([1, 4, 4])
tsn_ratio = tsn_ratio / np.sum(tsn_ratio)


def generate_random_flow(num_flow):
    """
    Generate random flow profiles.
    :param num_flow: the number of flows in the generated profile.
    :return: a numpy matrix describing the flow profile.
    """
    # Set the bounds for randomly generating flow profiles.
    rate_bound = (1, 100)
    burst_bound = (1, 100)
    # Configurations of deadline classes (uncomment the second line to switch from configuration 1 to 2).
    deadline_class = np.array([0.01, 0.1, 1])
    # deadline_class = np.array([0.01, 0.025, 0.05, 0.1])
    flow = np.zeros((num_flow, 3))
    rand_data = np.random.rand(num_flow, 2)
    # Randomly select the rate, burst, and deadline for each flow.
    # flow[:, 0] = np.around(rand_data[:, 0] * (rate_bound[1] - rate_bound[0]) + rate_bound[0], 2)
    # flow[:, 1] = np.around(rand_data[:, 1] * (burst_bound[1] - burst_bound[0]) + burst_bound[0], 2)

    rate_class = burst_class = np.array([5, 10, 20, 25, 40, 50, 80, 100])
    flow[:, 0] = rate_class[np.random.randint(len(rate_class), size=num_flow)]
    flow[:, 1] = burst_class[np.random.randint(len(burst_class), size=num_flow)]

    flow[:, 2] = deadline_class[np.random.randint(len(deadline_class), size=num_flow)]
    return flow


def generate_fb_flow(num_flow):
    """
    Generate random flow profiles using distribution reported in the Facebook paper.
    The paper is available at https://dl.acm.org/doi/10.1145/2785956.2787472.
    :param num_flow: the number of flows in the generated profile.
    :return: a numpy matrix describing the flow profile.
    """
    flow = np.zeros((num_flow, 3))
    int_num = len(cdf_values) - 1
    # Select the application type for each flow.
    flow_type = np.random.choice(3, num_flow, p=fb_ratio)
    # Randomly select the rate, burst, and deadline for each flow according to the Facebook flow distributions.
    int_point = np.random.randint(int_num, size=num_flow)
    size1, size2 = fb_size[flow_type, int_point], fb_size[flow_type, int_point + 1]
    duration1, duration2 = fb_duration[flow_type, int_point], fb_duration[flow_type, int_point + 1]
    rand_data = np.random.rand(2, num_flow)
    size = (size2 - size1) * rand_data[0] + size1
    duration = (duration2 - duration1) * rand_data[0] + duration1
    flow[:, 0] = size / duration
    flow[:, 1] = fb_burst[flow_type] * 2 * rand_data[1]
    flow[:, 2] = fb_deadline[flow_type]
    return flow


def generate_tsn_flow(num_flow, periodic=True):
    """
    Generate random flow profiles for TSN applications.
    Motivated by the following papers:
    https://ieeexplore.ieee.org/abstract/document/7092358/.
    https://ieeexplore.ieee.org/abstract/document/8700610/.
    https://ieeexplore.ieee.org/abstract/document/7385584/.
    :param num_flow: the number of flows in the generated profile.
    :param periodic: whether the arrival process is periodic with jitter or Poisson.
    :return: a numpy matrix describing the flow profile.
    """
    flow = np.zeros((num_flow, 3))
    # Sample traffic class according to the sampling ratio.
    class_mask = np.random.choice(len(tsn_ratio), num_flow, p=tsn_ratio)
    # Sample flow profiles for each traffic class.
    for class_idx in range(len(tsn_ratio)):
        num_flow = np.sum(class_mask == class_idx)
        flow_data = np.zeros((num_flow, 3))
        # Sample arrival interval from the specified traffic class.
        interval_mask = np.random.choice(len(tsn_interval[class_idx]), num_flow)
        # Set the rate, burst and deadline.
        flow_data[:, 0] = tsn_frame_size[class_idx] / tsn_interval[class_idx][interval_mask]
        flow_data[:, 1] = tsn_frame_size[class_idx]
        if not periodic:
            flow_data[:, 0] *= 1.1
            flow_data[:, 1] *= 25
        flow_data[:, 2] = tsn_deadline[class_idx]
        flow[class_mask == class_idx] = flow_data
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
    num_flow = 5
    np.random.seed(0)
    # flow_data = np.array([50, 50, 0.1]) * np.ones((num_flow, 3))
    # save_file(path, "flow1", flow_data)
    # path = f"../data/flow/tandem/rounded/{num_flow}/"
    # path = f"../data/flow/google/{num_flow}/"
    path = f"../data/flow/cev/{num_flow}/"
    path_route = f"../data/route/cev/{num_flow}/"
    for flow_idx in range(10):
        route_data = np.load(os.path.join(path_route, f"route{flow_idx + 1}.npy"))
        flow_num = len(route_data)
        flow_data = generate_tsn_flow(flow_num)
        # flow_data = generate_fb_flow(num_flow)
        save_file(path, f"flow{flow_idx + 1}", flow_data)
    # for file_idx in range(10):
    #     save_file(path, f"flow{file_idx + 1}", generate_random_flow(num_flow))
