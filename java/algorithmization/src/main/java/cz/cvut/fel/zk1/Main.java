package cz.cvut.fel.zk1;

import java.io.*;
import java.util.StringTokenizer;

@SuppressWarnings("Duplicates")
public class Main {

    private static final int CHAR_SIZE = 1 << 7;

    private int n, d, k, alphabetSize;
    private int ret;
    private Node root = new Node();

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
            alphabetSize = stringTokenizer.nextToken().length();
            n = Integer.parseInt(stringTokenizer.nextToken());
            d = Integer.parseInt(stringTokenizer.nextToken());
            k = Integer.parseInt(stringTokenizer.nextToken());
            for (int i = 0; i < n; i++) {
                line = br.readLine();
                stringTokenizer = new StringTokenizer(line);
                int l = Integer.parseInt(stringTokenizer.nextToken());
                Node curRoot = new Node();
                for (int j = 0; j < l; j++) {
                    line = br.readLine();
                    stringTokenizer = new StringTokenizer(line);
                    String string = stringTokenizer.nextToken();
                    insert(curRoot, string);
                }
                addDfs(curRoot, new StringBuilder());
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        dfs(root, 0, 0);
        System.out.println(ret);
    }

    void addDfs(Node rt, StringBuilder s) {
        for (int i = (int) 'a'; i <= (int) 'z'; ++i) {
            Node child = rt.children[i];
            if (child == null) {
                continue;
            }
            s.append((char) i);
            if (child.val > 0) {
                insert(root, s.toString());
            } else {
                addDfs(child, s);
            }
            s.deleteCharAt(s.length()-1);
        }
    }

    void dfs(Node n, int depth, int cumsum) {
        for (int i = (int) 'a'; i <= (int) 'z'; ++i) {
            Node child = n.children[i];
            if (child == null) {
                continue;
            }
            if (cumsum + child.val >= d) {
                ret += hugeModPow(k - depth - 1);
                ret %= 100_000;
            } else {
                dfs(child, depth + 1, cumsum + child.val);
            }
        }
    }

    int hugeModPow(int exp) {
        long leftVal = 1;
        long cur = alphabetSize;
        while (exp > 5) {
            if (exp % 2 == 0) {
                exp /= 2;
                cur = cur * cur % 100_000;
            } else {
                leftVal *= cur;
                leftVal %= 100_000;
                exp--;
                exp /= 2;
                cur = cur * cur % 100_000;
            }
        }
        long ret = 1;
        for (int i = 0; i < exp; i++) {
            ret *= cur;
            ret %= 100_000;
        }
        return (int) ((ret % 100_000 * leftVal % 100_000) % 100_000);
    }

    void insert(Node root, String s) {
        Node cur = root;
        int l = s.length();
        char[] chars = s.toCharArray();

        for (int i = 0; i < l; i++) {
            Node next = cur.children[chars[i]];
            if (next != null) {
                cur = next;
                continue;
            }
            for (; i < s.length(); i++) {
                Node newNode = new Node();
                cur.children[chars[i]] = newNode;
                cur = newNode;
            }
        }
        cur.val++;
    }

    class Node {

        int val;
        Node[] children;

        Node() {
            this.val = 0;
            this.children = new Node[CHAR_SIZE];
        }
    }
}

