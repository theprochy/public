package cz.cvut.fel.aic.zui.gobblet.algorithm;

import cz.cvut.fel.aic.zui.gobblet.environment.Board;

public abstract class Algorithm {
    private int counter = 0;

    public int run(Board game, int depth, final int player, int alpha, int beta) {
        // increment counter
        counter++;
        return runImplementation(game, depth, player, alpha, beta);
    }

    protected abstract int runImplementation(Board game, int depth, final int player, int alpha, int beta);

    public final int getNodesCount() {
        return counter;
    }
}
