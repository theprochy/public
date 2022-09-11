package cz.cvut.fel.aic.zui.gobblet;

import cz.cvut.fel.aic.zui.gobblet.algorithm.Algorithm;
import cz.cvut.fel.aic.zui.gobblet.environment.Board;
import cz.cvut.fel.aic.zui.gobblet.environment.Move;
import cz.cvut.student.AlphaBeta;

import java.util.ArrayList;
import java.util.Random;

import static cz.cvut.fel.aic.zui.gobblet.environment.Board.BLACK;
import static cz.cvut.fel.aic.zui.gobblet.environment.Board.WHITE_PLAYER;

public class Gobblet {


    public static void main(String[] args) {

        //game configuration
        //small
        //Value : -329
        // about 19000 nodes
        /*int seed = 0;
        int randomMoves = 12;
        int depth = 3;*/

        //bigger
        //Value : 659
        //about 4 700 000 nodes
        //int seed = 10;
        //int randomMoves = 10;
        //int depth = 5;


        //even bigger
        //Value : -41
        //about 19 000 000 nodes
        int seed = 20;
        int randomMoves = 12;
        int depth = 5;

        Board game = new Board();
        int playerToMove = randomPlay(game, randomMoves, seed);

        long currTime = System.currentTimeMillis();

        Algorithm ab = new AlphaBeta();

        System.out.println("Value : " + ab.run(game, depth, playerToMove, Integer.MIN_VALUE, Integer.MAX_VALUE));
        System.out.println("Nodes : " + ab.getNodesCount());

        currTime = System.currentTimeMillis() - currTime;
        System.out.println("Time : " + currTime / 1E6);

    }

    public static int randomPlay(Board game, int randomMoves, int randomSeed) {
        int playerToMove = WHITE_PLAYER;

        Random rnd = new Random(randomSeed);
        for (int i = 0; i < randomMoves; i++) {
            ArrayList<Move> successors = game.generatePossibleMoves(playerToMove);
            Move move = successors.get(rnd.nextInt(successors.size()));
            boolean move_made = game.makeMove(move);
            if (!move_made)
                System.err.println("Something is terribly wrong!");
            playerToMove = switchPlayer(playerToMove);
        }
        return playerToMove;
    }

    public static int switchPlayer(int player) {
        return (player == WHITE_PLAYER) ? BLACK : WHITE_PLAYER;
    }

}