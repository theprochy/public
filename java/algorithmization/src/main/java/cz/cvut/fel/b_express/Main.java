package cz.cvut.fel.b_express;

import java.io.*;
import java.util.*;

public class Main {

    private int n, t, count;
    private boolean[] visited;
    private int[] open;
    private int[] costs;
    private int[] lengths;
    private int[] componentIds;
    private int[] componentSizes;
    private int[] reverseNeighbors;
    private List<Set<Integer>> neighbors = new ArrayList<>();
    private Deque<Integer> stack;

    public static void main(String[] args) {
        new Main().run(null);
    }

    public void run(String file) {
        //parsing input
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        if (file != null) {
            try {
                br = new BufferedReader(new FileReader(new File(file)));
            } catch (FileNotFoundException e) {
                e.printStackTrace();
                System.exit(1);
            }
        }
        String line;
        StringTokenizer stringTokenizer;
        try {
            line = br.readLine();
            stringTokenizer = new StringTokenizer(line);
            n = Integer.parseInt(stringTokenizer.nextToken());
            int m = Integer.parseInt(stringTokenizer.nextToken());
            costs = new int[n];
            lengths = new int[n];
            componentSizes = new int[n];
            reverseNeighbors = new int[n];
            for (int i = 0; i < n; i++) {
                neighbors.add(new HashSet<>());
            }
            for (int i = 0; i < m; i++) {
                line = br.readLine();
                stringTokenizer = new StringTokenizer(line);
                int u = Integer.parseInt(stringTokenizer.nextToken());
                int v = Integer.parseInt(stringTokenizer.nextToken());
                neighbors.get(u).add(v);
                reverseNeighbors[v]++;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        tarjan();
        expressPaths();
    }

    private void expressPaths() {
        int maxCost = 0;
        int[] vertices_s = new int[n];
        for (int i = 0; i < n; i++) {
            costs[i] = componentSizes[componentIds[i]];
            if (costs[i] > maxCost) {
                maxCost = costs[i];
            }
        }

        int[] counts = new int[maxCost + 1];
        for (int i = 0; i < n; i++) {
            counts[costs[i]]++;
        }
        for(int i = 1; i < maxCost + 1; ++i) {
            counts[i] += counts[i - 1];
        }
        for (int i = 0; i < n; i++) {
            vertices_s[--counts[costs[i]]] = i;
        }
        Set<Integer> components = new HashSet<>();
        for (int i = 0; i < n; i++) {
            int node = vertices_s[i];
            if (costs[node] > componentSizeByNode(node)) {
                continue;
            }
            int nodeComponent = componentIds[node];
            components.add(nodeComponent);
            expressPath(node, components, costs, lengths);
            components.remove(nodeComponent);
        }

        maxCost = 0;
        int maxLength = 0;
        for (int i = 0; i < n; i++) {
            if (costs[i] > maxCost) {
                maxCost = costs[i];
                maxLength = lengths[i];
            } else if (costs[i] == maxCost && lengths[i] > maxLength) {
                maxLength = lengths[i];
            }
        }

        System.out.println(maxCost + " " + maxLength);
    }

    private void expressPath(int node, Set<Integer> components, int[] costs, int[] lengths) {
        for (Integer neighbor : neighbors.get(node)) {
            int neighborComponent = componentIds[neighbor];
            if (components.contains(neighborComponent)) {
                continue;
            }
            if (componentSizeByNode(neighbor) < componentSizeByNode(node)) {
                continue;
            }
            int newCost = costs[node] + componentSizeByNode(neighbor);
            if (newCost == costs[neighbor]) {
                if (lengths[node] + 1 > lengths[neighbor]) {
                    lengths[neighbor] = lengths[node] + 1;
                    components.add(neighborComponent);
                    expressPath(neighbor, components, costs, lengths);
                    components.remove(neighborComponent);
                }
            } else if (newCost > costs[neighbor]) {
                costs[neighbor] = newCost;
                lengths[neighbor] = lengths[node] + 1;
                components.add(neighborComponent);
                expressPath(neighbor, components, costs, lengths);
                components.remove(neighborComponent);
            }
        }
    }

    private int componentSizeByNode(int node) {
        return componentSizes[componentIds[node]];
    }

    private class NodeComparator implements Comparator<Integer> {
        @Override
        public int compare(Integer i1, Integer i2) {
            return Integer.compare(reverseNeighbors[i1], reverseNeighbors[i2]);
        }
    }

    private void tarjan() {
        open = new int[n];
        visited = new boolean[n];
        componentIds = new int[n];
        stack = new ArrayDeque<>();
        for (int v = 0; v < n; v++) {
            if (!visited[v]) {
                components(v);
            }
        }
    }

    private void components(int v) {
        stack.push(v);
        open[v] = t++;
        visited[v] = true;
        int min = open[v];
        for (int neighbors : neighbors.get(v)) {
            if (!visited[neighbors]) {
                components(neighbors);
            }
            if (open[neighbors] < min) {
                min = open[neighbors];
            }
        }
        if (min < open[v]) {
            open[v] = min;
            return;
        }
        int componentNode;
        do {
            componentNode = stack.pop();
            componentIds[componentNode] = count;
            componentSizes[count]++;
            open[componentNode] = n;
        } while (componentNode != v);
        count++;
    }
}
