package pal;

import java.io.*;
import java.util.*;

@SuppressWarnings("Duplicates")
public class Main {

    private int n, t, count = 1, dMax = 0;
    private boolean[] safe;
    private boolean[] visited;
    private int[] d;
    private int[] open;
    private int[] componentIds;
    private int[] currentComponent;
    private int[] componentSize;
    private int[] safeInComponent;
    private Set<Integer> startingComponents = new HashSet<>();
    private List<List<Integer>> neighbors;
    private List<List<Integer>> reverseNeighbors;
    private List<Set<Integer>> condensed = new ArrayList<>();
    private Deque<Integer> stack;

    public static void main(String[] args) {
        new Main().run(null);
    }

    public void run(String file) {
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
            neighbors = new ArrayList<>(n);
            reverseNeighbors = new ArrayList<>(n);
            safe = new boolean[n];
            currentComponent = new int[n];
            componentSize = new int[n + 1];
            safeInComponent = new int[n + 1];
            condensed.add(null);
            for (int i = 0; i < n; i++) {
                neighbors.add(new ArrayList<>());
                reverseNeighbors.add(new ArrayList<>());
            }
            for (int i = 0; i < m; i++) {
                line = br.readLine();
                stringTokenizer = new StringTokenizer(line);
                int u = Integer.parseInt(stringTokenizer.nextToken());
                int v = Integer.parseInt(stringTokenizer.nextToken());
                neighbors.get(u).add(v);
                reverseNeighbors.get(v).add(u);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        tarjan();
        longestTrips();
        System.out.println(dMax);
    }

    private void longestTrips() {
        d = new int[n];
        for (Integer start : startingComponents) {
            d[start] = safeInComponent[start];
        }
        for (int v = count - 1; v > 0; v--) {
            if (d[v] > dMax) {
                dMax = d[v];
            }
            for (Integer w : condensed.get(v)) {
                int component = componentIds[w];
                if (d[v] + safeInComponent[component] >= d[component]) {
                    d[component] = d[v] + safeInComponent[component];
                }
            }
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
                startingComponents.add(count - 1);
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
        int i = 0;
        Set<Integer> componentNeighbors = new HashSet<>();
        do {
            componentNode = stack.pop();
            currentComponent[i++] = componentNode;
            componentIds[componentNode] = count;
            componentSize[count]++;
            open[componentNode] = n;
        } while (componentNode != v);
        for (i = 0; i < componentSize[count]; i++) {
            boolean sf = true;
            for (Integer w : neighbors.get(currentComponent[i])) {
                if (componentIds[v] != componentIds[w]) {
                    sf = false;
                    componentNeighbors.add(w);
                }
            }
            for (Integer w : reverseNeighbors.get(currentComponent[i])) {
                if (componentIds[v] != componentIds[w]) {
                    sf = false;
                }
            }
            if (sf) {
                safe[v] = true;
                safeInComponent[count]++;
            }
        }
        count++;
        condensed.add(componentNeighbors);
    }
}
