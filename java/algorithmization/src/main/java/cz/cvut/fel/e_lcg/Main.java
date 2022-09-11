package cz.cvut.fel.e_lcg;

import java.io.*;
import java.util.*;

@SuppressWarnings("Duplicates")
public class Main {

    private int m, x2, x3;

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
            m = Integer.parseInt(stringTokenizer.nextToken());
            x2 = Integer.parseInt(stringTokenizer.nextToken());
            x3 = Integer.parseInt(stringTokenizer.nextToken());
        } catch (IOException e) {
            e.printStackTrace();
        }
        int l = Math.max((int)Math.round(Math.sqrt(m)), 30);
        boolean[] primes = new boolean[l];
        Set<Integer> primeDivisors = new HashSet<>();
        int prime = 2;
        for (int j = 2 * prime; j < l; j += prime) {
            primes[j] = true;
        }
        int mm = m;
        while (mm != 1) {
            if (mm % prime == 0) {
                mm /= prime;
                primeDivisors.add(prime);
            } else {
                prime++;
                while(primes[prime]) {
                    prime++;
                }
                for (int j = 2 * prime; j < l; j += prime) {
                    primes[j] = true;
                }
            }
        }

        long n = 0, min = Integer.MAX_VALUE, max = Integer.MIN_VALUE;

        outer:
        for (long a = 2; a <= m; ++a) {
            long c = (x3 - a * x2) % m;
            if (c < 0) { c += m; }
            if (c == 0 || c > 1 && m % c == 0) { continue; }
            if (m % 4 == 0 && (a - 1) % 4 != 0) { continue; }
            for (Integer divisor : primeDivisors) {
                if ((a - 1) % divisor != 0) { continue outer; }
            }
            long s = 0, old_s = 1;
            long t = 1, old_t = 0;
            long r = a, old_r = m;

            while (r != 0) {
                long q = old_r / r;
                long temp = r;
                r = old_r - q * r;
                old_r = temp;
                temp = s;
                s = old_s - q * s;
                old_s = temp;
                temp = t;
                t = old_t - q * t;
                old_t = temp;
            }
            long x1 = (x2 - c) * old_t % m;
            if (x1 < 0) { x1 += m; }
            if (x1 > max) { max = x1; }
            if (x1 < min) { min = x1; }
            n++;
            System.out.println(a + " " + c);
        }

        System.out.println(n + " " + min + " " + max);
    }
}
