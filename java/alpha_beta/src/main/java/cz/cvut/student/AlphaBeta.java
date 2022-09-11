package cz.cvut.student;

import cz.cvut.fel.aic.zui.gobblet.Gobblet;
import cz.cvut.fel.aic.zui.gobblet.algorithm.Algorithm;
import cz.cvut.fel.aic.zui.gobblet.environment.Board;
import cz.cvut.fel.aic.zui.gobblet.environment.Move;

import java.util.*;

@SuppressWarnings("Duplicates")
public class AlphaBeta extends Algorithm {

    private Map<Long, Integer> minMap = new HashMap<>();
    private Map<Long, Integer> maxMap = new HashMap<>();

    @Override
    protected int runImplementation(Board game, int depth, int player, int alpha, int beta) {

        if (depth == 0 || game.isTerminate(player) != -1)
            return game.evaluateBoard();

        ArrayList<Move> successors = game.generatePossibleMoves(player);

        PriorityQueue<BoardWrapper> pq;

        if(player == Board.WHITE_PLAYER) {
            pq = new PriorityQueue<>(new MaxComparator());
        } else {
            pq = new PriorityQueue<>(new MinComparator());
        }

        for (Move m : successors) {
            Board subGame = new Board(game);
            if(!subGame.makeMove(m))
                System.err.println("Something is terribly wrong in AB!");

            pq.add(new BoardWrapper(subGame));
        }

        int value;

        if(player == Board.WHITE_PLAYER) {
            long hash = game.calculateSimpleHash();

            if (maxMap.containsKey(hash)) {
                alpha = Integer.max(alpha, maxMap.get(hash));

                if (alpha >= beta) return alpha;
            }

            value = Integer.MIN_VALUE;

            while(!pq.isEmpty()) {
                Board subGame = pq.poll().board;

                int run_val = run(subGame, depth - 1, Gobblet.switchPlayer(player),
                        alpha, beta);
                value = Integer.max(value, run_val);

                alpha = Integer.max(alpha, value);
                if (alpha >= beta) break;
            }
            maxMap.put(game.calculateSimpleHash(), value);

        } else {
            long hash = game.calculateSimpleHash();

            if (minMap.containsKey(hash)) {
                beta = Integer.min(beta, minMap.get(hash));

                if (alpha >= beta) return beta;
            }

            value = Integer.MAX_VALUE;

            while (!pq.isEmpty()) {
                Board subGame = pq.poll().board;

                int run_val = run(subGame, depth - 1, Gobblet.switchPlayer(player),
                        alpha, beta);

                value = Integer.min(value, run_val);
                beta = Integer.min(beta, value);

                if (alpha >= beta) break;
            }
            minMap.put(game.calculateSimpleHash(), value);
        }

        return value;
    }

    private class BoardWrapper {
        private int value;
        private Board board;

        BoardWrapper(Board board) {
            this.board = board;
            this.value = board.evaluateBoard();
        }
    }

    private class MaxComparator implements Comparator<BoardWrapper> {
        @Override
        public int compare(BoardWrapper bw1, BoardWrapper bw2) {
            return Integer.compare(bw2.value, bw1.value);
        }
    }

    private class MinComparator implements Comparator<BoardWrapper> {

        @Override
        public int compare(BoardWrapper bw1, BoardWrapper bw2) {
            return Integer.compare(bw1.value, bw2.value);
        }
    }
}
