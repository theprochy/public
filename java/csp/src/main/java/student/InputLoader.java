package student;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

/**
 * Use this class to load the input and generate the CSP definition.
 */
class InputLoader {

    Problem load(InputStream in) throws IOException {

        BufferedReader reader = new BufferedReader(new InputStreamReader(in));
        String[] cnts = reader.readLine().split(",");
        int cntRows = Integer.parseInt(cnts[0]);
        int cntCols = Integer.parseInt(cnts[1]);

        Problem problem = new Problem(cntRows, cntCols);

        for( int i=0; i<cntRows; i++){
            String[] line = reader.readLine().split(",");
            problem.hcons.add(new ArrayList<Constr>());

            for (int j = 0; j < line.length; j++) {
                problem.hcons.get(i).add(new Constr(line[j].toCharArray()[0], Integer.parseInt(line[++j])));
            }
        }

        for( int i=0; i<cntCols; i++){
            String[] line = reader.readLine().split(",");
            problem.vcons.add(new ArrayList<Constr>());

            for (int j = 0; j < line.length; j++) {
                problem.vcons.get(i).add(new Constr(line[j].toCharArray()[0], Integer.parseInt(line[++j])));
            }
        }

        problem.generateDomains();

        return problem;
    }
}
