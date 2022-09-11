//package pal;
package cz.cvut.fel.a_mst;

import java.io.*;
import java.util.*;

@SuppressWarnings("Duplicates")
public class Main {

    private int r;
    private int c;
    private int n;
    private int k;
    private int[] d;
    private int[] phi;
    private int[] F;
    private boolean[] F_o;
    private boolean[] visited;

    private Queue<Integer> q;

    public static void main(String[] args) {
        new Main().run();
    }

    private void run() {
        //parsing input
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));

        try {
            br = new BufferedReader(new FileReader(new File("datapub/datapub1/pub11.in")));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            System.exit(1);
        }

        String line;
        StringTokenizer stringTokenizer;
        try {
            line = br.readLine();
            stringTokenizer = new StringTokenizer(line);
            r = Integer.parseInt(stringTokenizer.nextToken());
            c = Integer.parseInt(stringTokenizer.nextToken());
            n = r * c;
            int p = Integer.parseInt(stringTokenizer.nextToken());
            k = Integer.parseInt(stringTokenizer.nextToken());
            d = new int[n];
            phi = new int[n];
            F = new int[n];
            F_o = new boolean[n];
            visited = new boolean[n];
            q = new LinkedList<>();
            for (int i = 0; i < n; i++) {
                    phi[i] = Integer.MAX_VALUE;
                    d[i] = Integer.MAX_VALUE;
            }
            for (int i = 0; i < p; i++) {
                line = br.readLine();
                stringTokenizer = new StringTokenizer(line);
                int r1 = Integer.parseInt(stringTokenizer.nextToken()) - 1;
                int c1 = Integer.parseInt(stringTokenizer.nextToken()) - 1;
                int pot = Integer.parseInt(stringTokenizer.nextToken());
                d[r1 * c + c1] = 0;
                phi[r1 * c + c1] = pot;
                q.add(r1 * c + c1);
            }
            for (int i = 0; i < k; i++) {
                line = br.readLine();
                stringTokenizer = new StringTokenizer(line);
                int r1 = Integer.parseInt(stringTokenizer.nextToken())-1;
                int c1 = Integer.parseInt(stringTokenizer.nextToken())-1;
                int r2 = Integer.parseInt(stringTokenizer.nextToken())-1;
                int c2 = Integer.parseInt(stringTokenizer.nextToken())-1;
                F_o[r1 * c + c1] = true;
                F[r1 * c + c1] = r2 * c + c2 + 1;
                F[r2 * c + c2] = r1 * c + c1 + 1;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        // multilabel bfs starting in all nodes with nonzero potential
        while (!q.isEmpty()) {
            int idx = q.poll();
            if (visited[idx]) {
                continue;
            }
            visited[idx] = true;
            if (idx >= c && !visited[idx - c]) {
                potentialUpdate(idx - c, d[idx] + 1, phi[idx]);
            }
            if (idx % c > 0 && !visited[idx - 1]) {
                potentialUpdate(idx - 1, d[idx] + 1, phi[idx]);
            }
            if (idx < n - c && !visited[idx + c]) {
                potentialUpdate(idx + c, d[idx] + 1, phi[idx]);
            }
            if (idx % c < c - 1 && !visited[idx + 1]) {
                potentialUpdate(idx + 1, d[idx] + 1, phi[idx]);
            }
            if (F[idx] > 0 && !visited[F[idx] - 1]) {
                potentialUpdate(F[idx] - 1, d[idx] + 1, phi[idx]);
            }
        }

        // mst
        int max_w = -1;
        int m = 2 * r * c - r - c + k;
        int[] from = new int[m];
        int[] to = new int[m];
        int[] w = new int[m];
        int[] from_s = new int[m];
        int[] to_s = new int[m];
        int[] w_s = new int[m];
        int current_m = 0;

        for (int i = 0; i < n; ++i) {
            if (i % c < c - 1) {
                from[current_m] = i;
                to[current_m] = i + 1;
                w[current_m] = edgeWeight(i, i + 1);
                if (w[current_m] > max_w) {
                    max_w = w[current_m];
                }
                current_m++;
            }
            if (i < n - c) {
                from[current_m] = i;
                to[current_m] = i + c;
                w[current_m] = edgeWeight(i, i + c);
                if (w[current_m] > max_w) {
                    max_w = w[current_m];
                }
                current_m++;
            }
            if (F_o[i]) {
                from[current_m] = i;
                to[current_m] = F[i] - 1;
                w[current_m] = edgeWeight(i, F[i] - 1);
                if (w[current_m] > max_w) {
                    max_w = w[current_m];
                }
                current_m++;
            }
        }

        int[] counts = new int[max_w + 1];
        for (int i = 0; i < m; i++) {
            counts[w[i]]++;
        }
        for (int i = 1; i <= max_w; i++) {
            counts[i] += counts[i - 1];
        }
        for (int i = 0; i < m; i++) {
            int idx = --counts[w[i]];
            from_s[idx] = from[i];
            to_s[idx] = to[i];
            w_s[idx] = w[i];
        }

        int ret = 0, count = 0;
        int[] sets = new int[n];
        int[] parents = new int[n];
        for (int i = 0; i < n; i++) {
            sets[i] = i;
            parents[i] = -1;
        }

        for (int i = 0; count < n - 1 && i < m; i++) {
            int root1 = findRoot(from_s[i], parents);
            int root2 = findRoot(to_s[i], parents);
            if (sets[root1] != sets[root2]) {
                ret += w_s[i];
                parents[root2] = root1;
                count++;
            }
        }
        System.out.println(ret);
    }

    private int findRoot(int idx, int[] parents) {
        int parent = parents[idx];
        while (parent != -1) {
            idx = parent;
            parent = parents[idx];
        }
        return idx;
    }

    private void potentialUpdate(int idx, int d1, int phi1) {
        if (d[idx] > d1) {
            d[idx] = d1;
            phi[idx] = phi1;
            q.add(idx);
        } else if (d[idx] == d1) {
            if (phi[idx] > phi1) {
                phi[idx] = phi1;
            }
        }
    }

    private int edgeWeight(int idx1, int idx2) {
        return d[idx1] + d[idx2] + Math.abs(phi[idx1] - phi[idx2]);
    }
}
