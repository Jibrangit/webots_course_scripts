import typing as t
import numpy as np
from collections import deque, defaultdict
from heapq import heapify, heappush, heappop
import matplotlib.pyplot as plt


def get_neighbors(map: np.array, idx: t.Tuple):
    width = len(map)
    height = len(map[0])
    cell_cutoff = 0.3

    neighbor_indexes = [(0, 1), (0, -1), (-1, 0), (1, 0)]
    neighbors = []
    for n_idx in neighbor_indexes:
        n = (idx[0] + n_idx[0], idx[1] + n_idx[1])
        if 0 <= n[0] < width and 0 <= n[1] < height:
            if map[n]:
                neighbors.append(n)
    return neighbors


def get_diagonal_neighbors(map: np.array, idx: t.Tuple) -> t.List:
    """
    Returns neighbor indices along with costs to get to them from map[idx]
    """
    width = len(map)
    height = len(map[0])
    cell_cutoff = 0.3

    neighbor_indexes_costs = [
        (0, 1, 1),
        (0, -1, 1),
        (-1, 0, 1),
        (1, 0, 1),
        (1, 1, np.sqrt(2)),
        (-1, 1, np.sqrt(2)),
        (-1, -1, np.sqrt(2)),
        (1, -1, np.sqrt(2)),
    ]

    neighbors = []
    for n_idx in neighbor_indexes_costs:
        n = (idx[0] + n_idx[0], idx[1] + n_idx[1], n_idx[2])
        if 0 <= n[0] < width and 0 <= n[1] < height:
            if map[n[0], n[1]]:
                neighbors.append(n)
    return neighbors


def bfs(map: np.array, start: t.Tuple, goal: t.Tuple) -> t.List[t.Tuple]:
    visited = set()
    graph = {}
    q = deque()

    # cell : parent, cost
    graph[start] = list(start)
    q.append(start)

    # Use defaultdict for optimization.
    distances = {}
    for i in range(len(map)):
        for j in range(len(map[0])):
            distances[(i, j)] = float("inf")

    distances[start] = 0

    if not map[start]:
        print("Start position is non empty!")
        return []

    if not map[goal]:
        print("Goal position is non empty!")
        return []

    plt.imshow(map)  # shows the map
    plt.ion()

    while q:
        curr = q.popleft()

        if curr == goal:
            print("Goal reached!!!!")
            path = []
            while curr != start:
                path.append(curr)
                curr = tuple(graph[curr])

                plt.plot(curr[1], curr[0], "r*")  # puts a red asterisk at the goal
                plt.show()
                plt.pause(0.000001)

            path.reverse()
            return path

        else:
            neighbors = get_neighbors(map, curr)
            for neighbor in neighbors:
                cost = distances[curr] + 1

                if neighbor not in visited:
                    graph[neighbor] = list(curr)
                    distances[neighbor] = cost
                    visited.add(neighbor)

                elif neighbor in visited and cost < distances[neighbor]:
                    graph[neighbor] = list(curr)
                    distances[neighbor] = cost

                else:
                    continue

                q.append(neighbor)

        plt.plot(goal[1], goal[0], "y*")  # puts a yellow asterisk at the goal
        plt.plot(curr[1], curr[0], "g*")
        plt.show()
        plt.pause(0.000001)

    print("Path to goal could not be found!!")
    return []


def dijkstra(map: np.array, start: t.Tuple, goal: t.Tuple) -> t.List[t.Tuple]:
    visited = set()  # Set of Tuples
    graph = {}  # Dictionary of tuple (node) and list (parent)

    # Use defaultdict for optimization.
    distances = {}
    for i in range(len(map)):
        for j in range(len(map[0])):
            distances[(i, j)] = float("inf")

    distances[start] = 0

    q = []
    heapify(q)

    graph[start] = [list(start)]
    heappush(q, (distances[start], start))

    if not map[start]:
        print("Start position is non empty!")
        return []

    if not map[goal]:
        print("Goal position is non empty!")
        return []

    plt.imshow(map)  # shows the map
    plt.ion()

    while q:
        curr = heappop(q)  # (Distance from start, node)
        curr_node = curr[1]
        if curr_node == goal:
            print("Goal reached!!!!")
            path_node = curr[1]
            path = []
            while path_node != start:
                path.append(path_node)
                path_node = tuple(graph[path_node])

                plt.plot(
                    path_node[1], path_node[0], "r*"
                )  # puts a red asterisk at the goal
                plt.show()
                plt.pause(0.000001)

            path.reverse()
            return path

        else:
            neighbors = get_diagonal_neighbors(
                map, curr_node
            )  # List of Tuples[x, y, increment cost]
            for neighbor in neighbors:
                neighbor_idx = (neighbor[0], neighbor[1])
                cost = distances[curr_node] + neighbor[2]

                if (neighbor_idx) not in visited:
                    graph[neighbor_idx] = list(curr_node)
                    distances[neighbor_idx] = cost
                    visited.add(neighbor_idx)

                elif (neighbor_idx) in visited and cost < distances[neighbor_idx]:
                    graph[neighbor_idx] = list(curr_node)
                    distances[neighbor_idx] = cost

                else:
                    continue

                heappush(q, (distances[neighbor_idx], neighbor_idx))

        plt.plot(goal[1], goal[0], "y*")  # puts a yellow asterisk at the goal
        plt.plot(curr_node[1], curr_node[0], "g*")
        plt.show()
        plt.pause(0.000001)

    print("Path to goal could not be found!!")
    return []


def astar(map: np.array, start: t.Tuple, goal: t.Tuple) -> t.List[t.Tuple]:
    visited = set()  # Set of Tuples
    graph = {}  # Dictionary of tuple (node) and list (parent)

    # Use defaultdict for optimization.
    distances = {}
    for i in range(len(map)):
        for j in range(len(map[0])):
            distances[(i, j)] = float("inf")

    distances[start] = 0

    q = []
    heapify(q)

    graph[start] = [list(start)]
    heappush(
        q,
        (
            distances[start]
            + np.sqrt((goal[0] - start[0]) ** 2 + (goal[1] - start[1]) ** 2),
            start,
        ),
    )

    if not map[start]:
        print("Start position is non empty!")
        return []

    if not map[goal]:
        print("Goal position is non empty!")
        return []

    plt.imshow(map)  # shows the map
    plt.ion()

    while q:
        curr = heappop(q)  # (Distance from start, node)
        curr_node = curr[1]
        if curr_node == goal:
            print("Goal reached!!!!")
            path_node = curr[1]
            path = []
            while path_node != start:
                path.append(path_node)
                path_node = tuple(graph[path_node])

                plt.plot(
                    path_node[1], path_node[0], "r*"
                )  # puts a red asterisk at the goal
                plt.show()
                plt.pause(0.000001)

            path.reverse()
            return path

        else:
            neighbors = get_diagonal_neighbors(
                map, curr_node
            )  # List of Tuples[x, y, increment cost]
            for neighbor in neighbors:
                neighbor_idx = (neighbor[0], neighbor[1])
                cost = distances[curr_node] + neighbor[2]

                if (neighbor_idx) not in visited:
                    graph[neighbor_idx] = list(curr_node)
                    distances[neighbor_idx] = cost
                    visited.add(neighbor_idx)

                elif (neighbor_idx) in visited and cost < distances[neighbor_idx]:
                    graph[neighbor_idx] = list(curr_node)
                    distances[neighbor_idx] = cost

                else:
                    continue

                heappush(
                    q,
                    (
                        distances[neighbor_idx]
                        + np.sqrt(
                            (goal[0] - neighbor_idx[0]) ** 2
                            + (goal[1] - neighbor_idx[1]) ** 2
                        ),
                        neighbor_idx,
                    ),
                )

        plt.plot(goal[1], goal[0], "y*")  # puts a yellow asterisk at the goal
        plt.plot(curr_node[1], curr_node[0], "g*")
        plt.show()
        plt.pause(0.000001)

    print("Path to goal could not be found!!")
    return []


if __name__ == "__main__":
    rows = 20
    cols = 30
    np.random.seed(22)
    map = np.random.rand(rows, cols) < 0.9

    path = astar(map, (0, 0), (13, 15))
