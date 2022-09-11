package student;

import java.util.*;

/**
 * Use this class to hold your CSP definition: variables, domains, constraints. Pass this object wherever needed.
 */

class Problem {

    int r,c;

    String[] result;

    List<List<Constr>> hcons;
    List<List<Constr>> vcons;

    PriorityQueue<Var> hvars;
    List<Var> vvars;

    Problem(int r, int c) {
        this.r = r;
        this.c = c;
        result = new String[r];
        hcons = new ArrayList<>();
        vcons = new ArrayList<>();
        hvars = new PriorityQueue<>();
        vvars = new ArrayList<>();
    }

    void generateDomains() {
        for (int i = 0; i < r; i++) {
            List<Assignment> list = new ArrayList<>();
            generateDomain(list, hcons.get(i), "", c);
            hvars.add(new Var(i, true, list));
        }

        for (int i = 0; i < c; i++) {
            List<Assignment> list = new ArrayList<>();
            generateDomain(list, vcons.get(i), "", r);
            vvars.add(new Var(i, false, list));
        }
    }

    private void generateDomain(List<Assignment> results, List<Constr> constrs,
                                String assignment, int length) {

        int cur_length = assignment.length();

        if (constrs.isEmpty() && cur_length == length) { results.add(new Assignment(assignment)); }
        else {
            if (cur_length == length) return;

            StringBuilder new_assignment = new StringBuilder(assignment);

            assignment += '_';

            generateDomain(results, constrs, assignment, length);

            if (constrs.isEmpty()) return;
            if (cur_length > 0) {
                if (constrs.get(0).c == assignment.charAt(cur_length - 1)) return;
            }
            int new_length = cur_length + constrs.get(0).n;
            if (new_length > length) return;

            if (cur_length != 0) {
                if (assignment.charAt(cur_length) == constrs.get(0).c)
                    return;
            }

            List<Constr> new_constrs = new ArrayList<>(constrs);

            for (int i = 0; i < constrs.get(0).n; i++) {
                new_assignment.append(constrs.get(0).c);
            }
            new_constrs.remove(0);

            generateDomain(results, new_constrs, new_assignment.toString(), length);
        }
    }
}
