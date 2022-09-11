package pal;

import java.io.*;

public class Runner {

    public static void main(String[] args) {
        String file;
        BufferedReader br;
        int assignmentNumber = 7;
        int[] ins = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        long total = 0;
        for (int no : ins) {
            if (no < 10) {
                System.out.println("pub0" + no);
                file = "../datapub/datapub" + assignmentNumber + "/pub0" + no;
            } else {
                System.out.println("pub" + no);
                file = "../datapub/datapub" + assignmentNumber + "/pub" + no;
            }
            System.out.print("mine: ");
            long startTime = System.currentTimeMillis();
            new Main().run(file + ".in");
            long time = System.currentTimeMillis() - startTime;
            total += time;
            try {
                br = new BufferedReader(new FileReader(new File(file + ".out")));
                System.out.println("real: " + br.readLine());
            } catch (Exception e) {
                e.printStackTrace();
            }
            System.out.println(time + " ms");
        }
        System.out.println("Total ms: " + total);
    }
}
