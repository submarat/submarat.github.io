A recent coding interview question I saw.

Problem: A currency converter has the exchange rates exchange, such that exchange[i][j] represents the amount of money you would get for exchanging 1 unit of the ith currency for 1 unit of the jth currency. A "non-exchange" (that is, exchanging a currency with itself) is represented by exchange[i][i] = 1. Try to find solution in n^3 time.

Example:
Input: [[1, 0.5, 0.2], [1.8, 1, 0.5], [4.05, 1.2, 1]]
Output: true - we can exchange between currencies 1st, 2nd, 3rd and back to 1st to obtain a larger amount.

## Solution 1: Explore all paths using DFS

If we imagine each currency as a node in a directed, weighted graph, we can restate the problem as: expore all cycles from each node to itself and test whether we can arrive at a higher value. As we traverse each path, accumulate a product to test once the source node is seen.

We don't need to do this for each node because the final product will contain the costs of all the nodes in the cycle hence we can start from just one node.
To explore all paths we can use a modified version of DFS that keeps a list of visited edges.

This algorithm does the trick but has a runtime of n!. The dfs() function is called n times first, each time it recurses, it is called n-1 times, then n-2, n-3. Giving the runtime of O((n*(n-1)*(n-2)...(1)) or O(n!).

```
boolean currencyArbitrage(double[][] g) {
    if (g.length < 2) return false;
    
    boolean[][] visited = new boolean[g.length][g.length];
    
    for (int j = 0; j < g.length; j++) {
        if (dfs(g, j, g[0][j], visited)) {
            return true;
        }
    }
    return false;
}


boolean dfs(double[][] g, int i, double p, boolean[][] visited) {
    if (i == 0) {
        return p > 1;
    }
    
    for (int j = 0; j < g.length; j++) {
        if (i != j && !visited[i][j]) {
            visited[i][j] = true;
            if (dfs(g, j, p*g[i][j], visited)) {
                return true;
            }
            visited[i][j] = false;
        }
    }
    return false;
}
```

## Solution 2: Floyd-Warshall algorithm

We can do better than O(n!) by modifying FW. The original algorithm finds all (I know, right!) shortest paths in a directed weighted graph in O(n^3) time by incrementally updating its estimates. It does so by choosing the cheapest of two options: a) traverse the cheapest known path (i, j) b) traverse path between (i, k) + (k, j). All we need to do to adapt the solution is to choose the more expensive option to find the longest paths and to update the cost of path (i, k) + (k, j) by multiplying their respective costs.

The runtime is inherited from FW - O(n^3).

```
boolean currencyArbitrage(double[][] g) {
    if (g.length < 2) return false;
    
    for (int k = 0; k < g.length; k++) {
        for (int i = 0; i < g.length; i++) {
            for (int j = 0; j < g.length; j++) {
                double d = g[i][k] * g[k][j];
                if (g[i][j] < d) {
                    g[i][j] = d;
                }
            }
        }
    }
    return g[0][0] > 1;
}
```
