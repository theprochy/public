package student;

import java.io.IOException;
import java.util.List;

/**
 * This is the entry point of the program.
 *
 * This structure is to help you with programming. If you need to slightly change some definitions, don't be afraid to do so.
 *
 */
public class CSPMain {
    public static void main(String[] args) {
        // load the input
        Problem problem;

        try {
            InputLoader loader = new InputLoader();
            problem = loader.load(System.in);
            System.out.println();
        }
        catch(IOException exception) {
            exception.printStackTrace();
            return;
        }

        // run the solver
        Solver solver = new Solver();

        long start = System.currentTimeMillis();
        //System.out.println(Calendar.getInstance().get(Calendar.HOUR_OF_DAY) + ":" +
        //        Calendar.getInstance().get(Calendar.MINUTE));

        List<String[]> result = solver.solve(problem);

        if(result.isEmpty()) {
            System.out.println("null");
        } else {
            for (String[] sol : result) {
                for (String s : sol) {
                    System.out.println(s);
                }
                System.out.println();
            }
        }

        //System.out.println(System.currentTimeMillis() - start + " ms");
    }
}
