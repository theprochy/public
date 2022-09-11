package student;

import java.util.*;

/**
 * I chose rows and columns as variables. Firstly, I generate all possible assignments for each
 * column and row based on given constraints. This means, that when we decide to choose an assignment
 * for a row or column, there must exist at least one assignment in each orthogonal line/column, where
 * the character at the pixel, where they intersect is the same. I only run the ac3 algorithm
 * at the beginning and also use the most constrained variable heuristic.
 * However, I have not been able to tune my solution enough to solve the dino.txt,
 * probably my backtrack is slow. I spent a lot of time trying to pinpoint what is slowing me
 * down the most, but I have not been able to succeed.
 */

@SuppressWarnings("Duplicates")
class Solver {

    private int depth = 0;
    private int count = 0;

    private Var[] vars;
    private Problem p;
    private List<Var> list = new ArrayList<>();
    private List<String[]> solutions;

    List<String[]> solve(Problem problem){

        solutions = new ArrayList<>();
        p = problem;

        if(!ac3(p, null)) return solutions;

        Object[] objects = p.hvars.toArray();
        vars = new Var[objects.length];
        for (int i = 0; i < objects.length; i++) {
            vars[i] = (Var) objects[i];
        }
        Arrays.sort(vars);

        list.addAll(Arrays.asList(vars));

        backtrack();

        return solutions;
    }

    private void backtrack() {

        if (depth == p.r) {
            //System.out.println("solution found");
            String[] solution = new String[p.r];

            System.arraycopy(p.result, 0, solution, 0, p.r);

            solutions.add(solution);
            return;
        }

        Var x = list.get(depth);

        for (Assignment assignment : x.domain) {

            if(!assignment.valid) continue;
            String s = assignment.value;

            //if(depth == 0) System.out.println(count++);

            p.result[x.i] = s;

            boolean valid = true;

            List<Integer> var_indexes = new ArrayList<>();
            List<Integer> dom_indexes = new ArrayList<>();

            for (Var y : p.vvars) {

                boolean empty = true;

                for (int i = 0; i < y.domain.size(); ++i) {
                    Assignment a = y.domain.get(i);
                    if (a.valid) {
                        if (a.value.charAt(x.i) != s.charAt(y.i)) {
                            a.valid = false;
                            var_indexes.add(y.i);
                            dom_indexes.add(i);
                        } else {
                            empty = false;
                        }
                    }
                }

                if (empty) {
                    valid = false;
                    break;
                }
            }

            if (valid) {
                depth++;
                backtrack();
                depth--;
            }

            for(int i = var_indexes.size() - 1; i >= 0; i--) {
                p.vvars.get(var_indexes.get(i)).domain.get(dom_indexes.get(i)).valid = true;
            }
        }
    }

    /**
     * The Arc Consistency 3 algorithm
     *
     * This algorthim clears the variable domains of values, that would lead to
     * assignments that would give no solution. It works with a queue containing
     * variables to be revised. If the domain of a variable has been revised, it adds
     * all the arcs outcoming from the variable to the queue. In our case this simply
     * means that when the variable is horizontal, we add all vertical ones and
     * vice versa.
     *
     * @param p Problem on which the AC3 is to be run
     * @return false if the given problem has no solution, true otherwise
     */

    private boolean ac3(Problem p, List<Var> unsetVars) {
        //long start = System.currentTimeMillis();

        Collection<Var> hvars = unsetVars == null ? p.hvars : unsetVars;
        Set<Var> q = new LinkedHashSet<>(hvars);

        while (!q.isEmpty()) {
            Var x = q.iterator().next();
            q.remove(x);

            if (revise(p, x)) {
                if (x.domain.isEmpty()) {
                    //System.out.println("AC3 done in: " + (System.currentTimeMillis() - start) + " ms");
                    return false;
                }
                else {
                    q.addAll(x.isHorizontal ? p.vvars : hvars);
                }
            }
        }

        //System.out.println("AC3 done in: " + (System.currentTimeMillis() - start) + " ms");
        return true;
    }

    /**
     *  Check arc consistency for all arcs outcoming from Var X, that is, for arcs (X, Y)
     *  where Y are all horizontal or verical Vars depending on whether X is vertical
     *  or horizontal.
     *
     * @param p Problem which the revision is connected to
     * @param x Var to be revised
     * @return true if some element of X.domain was deleted, false otherwise
     */

    private boolean revise(Problem p, Var x) {

        Set<Assignment> toRemove = new HashSet<>();

        for (Assignment a : x.domain) {
            boolean validForAll = true;

            for (Var y : x.isHorizontal ? p.vvars : p.hvars) {
                boolean validForOne = false;

                for (Assignment aa : y.domain) {
                    if (a.value.charAt(y.i) == aa.value.charAt(x.i)) {
                        validForOne = true;
                        break;
                    }
                }

                if (!validForOne) {
                    validForAll = false;
                    break;
                }
            }

            if (!validForAll) {
                toRemove.add(a);
            }
        }

        x.domain.removeAll(toRemove);

        return !toRemove.isEmpty();
    }
}
