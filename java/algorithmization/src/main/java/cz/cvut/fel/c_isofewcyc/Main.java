package cz.cvut.fel.c_isofewcyc;

import java.io.*;
import java.util.*;

public class Main {

    private int[] depths;
    private Map<Integer, Long> hashByNode = new HashMap<>();
    private List<List<List<Integer>>> graphs = new ArrayList<>();

    public static void main(String[] args) {
        new cz.cvut.fel.d_geneng.Main().run(null);
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
        int N = 0;
        String line;
        StringTokenizer stringTokenizer;
        try {
            line = br.readLine();
            stringTokenizer = new StringTokenizer(line);
            int t = Integer.parseInt(stringTokenizer.nextToken());
            N = Integer.parseInt(stringTokenizer.nextToken());
            int m = Integer.parseInt(stringTokenizer.nextToken());
            depths = new int[N];
            for (int i = 0; i < t; i++) {
                List<List<Integer>> graph = new ArrayList<>();
                for (int j = 0; j < N; j++) {
                    graph.add(new ArrayList<>());
                }
                for (int j = 0; j < m; j++) {
                    line = br.readLine();
                    stringTokenizer = new StringTokenizer(line);
                    int u = Integer.parseInt(stringTokenizer.nextToken()) - 1;
                    int v = Integer.parseInt(stringTokenizer.nextToken()) - 1;
                    graph.get(u).add(v);
                    graph.get(v).add(u);
                }
                graphs.add(graph);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        Map<Long, Integer> hashCounts = new HashMap<>();
        for (List<List<Integer>> graph : graphs) {
            for (int i = 0; i < N; i++) {
                if (graph.get(i).size() > 1) {
                    Deque<Integer> s = new ArrayDeque<>();
                    findRoot(graph, i, new HashSet<>(), s, new HashSet<>());
                    hashCounts.merge(hashGraph(graph, s.pop(), new ArrayDeque<>(), new HashSet<>()), 1, Integer::sum);
                    hashByNode.clear();
                    break;
                }
            }
        }
        hashCounts.values().stream().sorted().forEach(i -> System.out.print(i + " "));
        System.out.println();
    }

    private long hashGraph(List<List<Integer>> graph, int v, Deque<Integer> s, Set<Integer> sSet) {
        s.push(v);
        sSet.add(v);
        long hash = 0;
        for (Integer w : graph.get(v)) {
            hashFromNode(graph, w, 1, s, sSet);
            hash += hashByNode.get(v);
        }
        return hash;
    }

    private void hashFromNode(List<List<Integer>> graph, int v, int depth, Deque<Integer> s, Set<Integer> sSet) {
        int prev = s.peek() == null ? -1 : s.peek();
        if (graph.get(v).size() == 1) {
            hashByNode.put(prev, 1L);
            return;
        }
        depths[v] = depth;
        s.push(v);
        sSet.add(v);
        for (Integer w : graph.get(v)) {
            if (!sSet.contains(v)) {
                break;
            }
            if (w.equals(prev) || hashByNode.containsKey(w)) {
                continue;
            } else if (sSet.contains(w)) {
                int x = 0, x_prev = -1;
                int l = depth - depths[w] + 1;
                long hash = 0;
                for (int i = 1; i < l; i++) {
                    int c = Math.min(i, l - i);
                    if (i > 1) {
                        x_prev = x;
                    }
                    x = s.peek();
                    if (hashByNode.containsKey(x)) {
                        hash += c * hashByNode.get(x);
                    } else {
                        for (Integer next : graph.get(x)) {
                            if (sSet.contains(next) || next == x_prev) {
                                continue;
                            }
                            hashFromNode(graph, next, depth - i + 2, s, sSet);
                            hash += c * hashByNode.get(x);
                        }
                    }
                    s.pop();
                    sSet.remove(x);
                }
                sSet.remove(s.pop());
                hashByNode.put(s.peek(), hash * l * 5678991L);
            } else {
                hashFromNode(graph, w, depth + 1, s, sSet);
            }
        }
    }

    private boolean findRoot(List<List<Integer>> graph, int v, Set<Integer> closed, Deque<Integer> s, Set<Integer> sSet) {
        int prev = s.peek() == null ? -1 : s.peek();
        s.push(v);
        sSet.add(v);
        for (Integer w : graph.get(v)) {
            if (w.equals(prev) || graph.get(w).size() == 1 || closed.contains(w)) {
                continue;
            } else if (sSet.contains(w)) {
                int x;
                do {
                    x = s.peek();
                    for (Integer next : graph.get(x)) {
                        if (graph.get(next).size() == 1 || sSet.contains(next) || closed.contains(next)) {
                            continue;
                        }
                        if (findRoot(graph, next, closed, s, sSet)) {
                            return true;
                        }
                    }
                    s.pop();
                    sSet.remove(x);
                    closed.add(x);
                } while (x != w);
            } else {
                if (findRoot(graph, w, closed, s, sSet)) {
                    return true;
                }
            }
        }
        if (s.peek() == v) {
            return true;
        }
        return false;
    }
}
