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
    distances={}
    for i in range(len(map)):
        for j in range(len(map[0])):
            distances[(i, j)] = float("inf")

    distances[start]=0

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
    graph = {}  # Dictionary of tuple keys and list of tuples (undirected connected nodes)

    # Use defaultdict for optimization. 
    distances={}
    for i in range(len(map)):
        for j in range(len(map[0])):
            distances[(i, j)] = float("inf")

    distances[start]=0

    q = []
    heapify(q)

    graph[start] = [list(start)]
    heappush(q, start)

    if not map[start]:
        print("Start position is non empty!")
        return []

    if not map[goal]:
        print("Goal position is non empty!")
        return []

    plt.imshow(map)  # shows the map
    plt.ion()

    while q:
        curr = heappop(q)

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
            neighbors = get_diagonal_neighbors(map, curr)   # List of Tuples[x, y, increment cost]
            for neighbor in neighbors:
                neighbor_idx = (neighbor[0], neighbor[1])
                cost = distances[curr] + neighbor[2]

                if (neighbor_idx) not in visited:
                    graph[neighbor_idx] = list(curr)
                    distances[neighbor_idx] = cost
                    visited.add(neighbor_idx)

                elif (neighbor_idx) in visited and cost < distances[neighbor_idx]:
                    graph[neighbor_idx] = list(curr)
                    distances[neighbor_idx] = cost 

                else:
                    continue

                heappush(q, neighbor_idx)

        plt.plot(goal[1], goal[0], "y*")  # puts a yellow asterisk at the goal
        plt.plot(curr[1], curr[0], "g*")
        plt.show()
        plt.pause(0.000001)

    print("Path to goal could not be found!!")
    return []


if __name__ == "__main__":
    rows = 20
    cols = 30
    np.random.seed(21)
    map = np.random.rand(rows, cols) < 0.7

    path = dijkstra(map, (0, 0), (13, 15))
