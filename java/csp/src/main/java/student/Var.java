package student;


import java.util.List;

class Var implements Comparable {

    boolean isHorizontal;
    int i;

    List<Assignment> domain;

    Var (int i, boolean isHorizontal, List<Assignment> domain) {
        this.i = i;
        this.domain = domain;
        this.isHorizontal = isHorizontal;
    }

    @Override
    public int compareTo(Object o) {
        return Integer.compare(this.domain.size(), ((Var) o).domain.size());
    }
}
