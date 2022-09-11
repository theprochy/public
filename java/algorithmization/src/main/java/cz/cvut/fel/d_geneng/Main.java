package cz.cvut.fel.d_geneng;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

@SuppressWarnings("Duplicates")
public class Main {

    private int dMax;
    private String goal;
    private List<String> basesSorted = new ArrayList<>();
    private Map<String, Integer> bases = new HashMap<>();
    private Map<String, Integer> costs = new HashMap<>();
    private Map<String, Integer> lengths = new HashMap<>();

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
            goal = stringTokenizer.nextToken();
            line = br.readLine();
            stringTokenizer = new StringTokenizer(line);
            int n = Integer.parseInt(stringTokenizer.nextToken());
            dMax = Integer.parseInt(stringTokenizer.nextToken());

            for (int i = 0; i < n; i++) {
                line = br.readLine();
                stringTokenizer = new StringTokenizer(line);
                int cost = Integer.parseInt(stringTokenizer.nextToken());
                line = br.readLine();
                stringTokenizer = new StringTokenizer(line);
                bases.merge(stringTokenizer.nextToken(), cost, Math::min);
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
        costs.put("", 0);
        lengths.put("", 0);
        basesSorted = bases.keySet().stream().sorted(Comparator.comparingInt(b -> bases.get(b))).collect(Collectors.toList());
        sequence(goal);
        System.out.println(costs.get(goal) + " " + lengths.get(goal));
    }

    private void sequence(String s) {
        int minCost = Integer.MAX_VALUE;
        int minLength = Integer.MAX_VALUE;
        for (String base : basesSorted) {
            String match = match(base, s);
            if (!match.isEmpty()) {
                String newS = s.substring(match.length());
                if (!newS.isEmpty() && !costs.containsKey(newS)) {
                    sequence(newS);
                }
                if (costs.get(newS) != Integer.MAX_VALUE) {
                    int newCost = bases.get(base) + base.length() - match.length() + costs.get(newS);
                    int newLength = 1 + lengths.get(newS);
                    if (newCost < minCost) {
                        minCost = newCost;
                        minLength = newLength;
                    } else if (newCost == minCost) {
                        if (newLength < minLength) {
                            minLength = newLength;
                        }
                    }
                }
            }
        }
        costs.put(s, minCost);
        lengths.put(s, minLength);
    }

    private String match(String base, String goal) {
        StringBuilder builder = new StringBuilder();
        int deletions = 0;
        char[] baseChar = base.toCharArray();
        char[] goalChar = goal.toCharArray();
        for (int i = 0, j = 0; i < baseChar.length && j < goal.length(); ++i) {
            if (deletions > dMax) {
                return "";
            }
            if (baseChar[i] == goalChar[j]) {
                builder.append(baseChar[i]);
                j++;
            } else {
                deletions++;
            }
        }
        return builder.toString();
    }
}
