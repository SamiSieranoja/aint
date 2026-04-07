import math
import matplotlib.pyplot as plt
import random


def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def tsp_closed_tour_cost(cities, tour):
    total_cost = 0.0
    edge_costs = []

    n = len(tour)
    for i in range(n):
        a = tour[i]
        b = tour[(i + 1) % n]  # closed loop
        dist = euclidean_distance(cities[a], cities[b])
        total_cost += dist
        edge_costs.append(((a, b), dist))

    return total_cost, edge_costs


def visualize_tsp(cities, tour, total_cost, edge_costs, live=False):
    if live:
        plt.clf()
    else:
        plt.figure(figsize=(8, 6))

    xs = [p[0] for p in cities]
    ys = [p[1] for p in cities]
    plt.scatter(xs, ys, s=100)

    for i, (x, y) in enumerate(cities):
        plt.text(x + 0.12, y + 0.12, f"{i}", fontsize=11)

    route_x = [cities[i][0] for i in tour] + [cities[tour[0]][0]]
    route_y = [cities[i][1] for i in tour] + [cities[tour[0]][1]]
    plt.plot(route_x, route_y, marker='o')

    for order, city_idx in enumerate(tour):
        x, y = cities[city_idx]
        plt.text(x - 0.35, y - 0.4, f"{order}", fontsize=9)

    for ((a, b), dist) in edge_costs:
        x_mid = (cities[a][0] + cities[b][0]) / 2
        y_mid = (cities[a][1] + cities[b][1]) / 2
        plt.text(x_mid, y_mid, f"{dist:.2f}", fontsize=8)

    plt.title(f"Closed-loop TSP ({len(cities)} cities)\nTotal cost = {total_cost:.3f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()

    if live:
        plt.pause(0.3)
    else:
        plt.show()
        
def main():
    # 4 outer hull corners + 4 interior points near opposite edges.
    # Optimal tour alternates outer<->inner (star pattern), which is
    # non-obvious and hard to find by random swaps alone.
    cities = [
        (0,  1.2),  
        (9.2, 0),  
        (9.8, 8.4), 
        (0.3,  10), 
        (5.5,  2),  
        (8,  5.8),  
        (4.2,  7.4),  
        (1.7,  4.5),  
        (5,  5.1),  
    ]

    tour = [1, 4, 5, 3, 7, 6, 4, 0, 2, 8]
    tour_best = tour
    cost_best, _ = tsp_closed_tour_cost(cities, tour)

    # hint:
    # a = random.randrange(0,len(cities)-1)
    # ...
            
    tour = tour_best
    total_cost, edge_costs = tsp_closed_tour_cost(cities, tour)
    

    print("Cities:")
    for i, p in enumerate(cities):
        print(f"  {i}: {p}")

    print("\nTour:")
    print("  " + " -> ".join(map(str, tour)) + f" -> {tour[0]}")

    print("\nEdge costs:")
    for (a, b), dist in edge_costs:
        print(f"  {a} -> {b}: {dist:.3f}")

    print(f"\nTotal closed-loop TSP cost = {total_cost:.3f}")
    
    visualize_tsp(cities, tour, total_cost, edge_costs)



if __name__ == "__main__":
    main()
