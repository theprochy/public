package cz.cvut.fel.aic.zui.gobblet.environment;

import java.util.ArrayList;
import java.util.HashMap;

public class Board {
    public static final int WHITE_PLAYER = 0;
    public static final int BLACK = 4;

    public static final int BOARD_SIZE = 4;
    public static final int PIECES_COUNT = 4;
    public static final int UNUSED_SETS = 3;
    public static final int W_UNUSED_ROW = BOARD_SIZE;
    public static final int B_UNUSED_COLUMN = BOARD_SIZE;

    private int[][] tiles = new int[BOARD_SIZE + 1][BOARD_SIZE + 1];

    public static final int WHITE_MASK = 15;
    public static final int BLACK_MASK = 15 << BLACK;

    public static final int[] PIECES = {1, 1 << 1, 1 << 2, 1 << 3};

    public static final int DUMMY = -1;
    public static final int DRAW = -2;

    public boolean draw = false;

    protected HashMap<Long, Integer> positionsCounter = new HashMap<Long, Integer>();

    /**
     * Method generates an unsorted list of currently possible moves.
     *
     * @param player to move
     * @return list of possible moves
     */
    public ArrayList<Move> generatePossibleMoves(int player) {
        ArrayList<Move> result = new ArrayList<Move>();

        int[][] fromPieces = new int[BOARD_SIZE + 1][BOARD_SIZE + 1];
        int[][] toPieces = new int[BOARD_SIZE][BOARD_SIZE];

        for (int i = 0; i < BOARD_SIZE + 1; i++) {
            for (int j = 0; j < BOARD_SIZE + 1; j++) {
                fromPieces[i][j] = getMovablePiece(i, j, player);
                if ((i < BOARD_SIZE) && (j < BOARD_SIZE)) {
                    toPieces[i][j] = getPlaceablePieces(i, j);
                }
            }
        }

        for (int i = 0; i < BOARD_SIZE + 1; i++) {
            for (int j = 0; j < BOARD_SIZE + 1; j++) {
                for (int k = 0; k < BOARD_SIZE; k++) {
                    for (int l = 0; l < BOARD_SIZE; l++) {
                        if ((fromPieces[i][j] & toPieces[k][l]) > 0) {
                            result.add(new Move(i, j, k, l, fromPieces[i][j], player));
                        }
                    }
                }
            }
        }

        return result;
    }

    /**
     * Method executes a move on the board.
     *
     * @param m - Move to execute.
     * @return FALSE in case the move is illegal, TRUE otherwise.
     */
    public boolean makeMove(Move m) {
        return makeMove(m.getFrom_x(), m.getFrom_y(), m.getTo_x(), m.getTo_y(), m.getPiece(), m.getPlayer());
    }

    /**
     * Method executes a move on the board.
     *
     * @param - Move to execute.
     * @return FALSE in case the move is illegal, TRUE otherwise.
     */
    protected final boolean makeMove(int from_x, int from_y, int to_x, int to_y, int piece, int player) {
        if (piece != getMovablePiece(from_x, from_y, player)) return false;
        if (!isPiecePlaceable(to_x, to_y, piece, player)) return false;

        //remove from old tile
        tiles[from_x][from_y] &= ~(piece << player);

        //put to a new tile
        tiles[to_x][to_y] |= (piece << player);

        // it adds the new board to the history
        int count = 1;
        long simpleHash = calculateSimpleHash();
        if (positionsCounter.get(simpleHash) != null) {
            count += positionsCounter.get(simpleHash);
        }
        positionsCounter.put(simpleHash, count);

        // draw check
        if (count >= 3) draw = true;

        return true;
    }

    /**
     * Initialize the board for a new game.
     */
    protected final void init() {
        for (int i = 0; i < UNUSED_SETS; i++) {
            for (int p = 0; p < PIECES_COUNT; p++) {
                tiles[W_UNUSED_ROW][i] |= PIECES[p] << WHITE_PLAYER;
                tiles[i][B_UNUSED_COLUMN] |= PIECES[p] << BLACK;
            }
        }
    }

    /**
     * Default constructor. It sets the board into the initial position.
     */
    public Board() {
        init();
    }

    /**
     * Copy constructor.
     *
     * @param b - Other board.
     */
    public Board(Board b) {
        for (int i = 0; i < BOARD_SIZE + 1; i++) {
            for (int j = 0; j < BOARD_SIZE + 1; j++) {
                this.tiles[i][j] = b.tiles[i][j];
            }
        }

        for (Long l : b.positionsCounter.keySet()) {
            this.positionsCounter.put(l, b.positionsCounter.get(l));
        }
    }


    @Override
    public String toString() {
        StringBuffer sb = new StringBuffer();
        sb.append("-----------------------------------------------------------------\n");
        for (int x = BOARD_SIZE - 1; x >= 0; x--) {
            sb.append(((char) (65 + x)) + " |\t");
            for (int y = 0; y < BOARD_SIZE; y++) {
                sb.append(tiles[x][y] + "\t|\t");
            }
            sb.append("\n");
        }
        sb.append("-----------------------------------------------------------------\n");
        sb.append("  |\t");
        for (int i = 0; i < BOARD_SIZE; i++) {
            sb.append(i + "\t|\t");
        }
        sb.append("\n\nW: ");
        for (int i = 0; i < UNUSED_SETS; i++) {
            sb.append(tiles[W_UNUSED_ROW][i] + " ");
        }
        sb.append("\nB: ");
        for (int i = 0; i < UNUSED_SETS; i++) {
            sb.append(tiles[i][B_UNUSED_COLUMN] + " ");
        }

        return sb.toString();
    }

    /**
     * Evaluates a single tile of the board.
     *
     * @param x - row of the tile
     * @param y - column of the tile
     * @return evaluation
     */
    protected final int evaluateTile(int x, int y) {
        int white = 0;
        int black = 0;

        int weak_tile = (((x - y) == 0) || ((x + y) == BOARD_SIZE - 1)) ? 2 : 1;

        int w_tile = tiles[x][y] & WHITE_MASK >> WHITE_PLAYER;
        int b_tile = (tiles[x][y] & BLACK_MASK) >> BLACK;

        int stones = 0;

        for (int i = PIECES_COUNT - 1; i >= 0; i--) {
            white += ((w_tile & PIECES[i]) >> i) * (i + 1) * (1 - stones / PIECES_COUNT);
            black += ((b_tile & PIECES[i]) >> i) * (i + 1) * (1 - stones / PIECES_COUNT);
            stones += ((w_tile & PIECES[i]) >> i) + ((b_tile & PIECES[i]) >> i);
        }

        return (white - black) * weak_tile;
    }

    /**
     * Evaluates current position on the board. Returns a value that the white (first) player
     * wants to maximize. I.e. the higher value, the better position for the white (first) player.
     *
     * @return evaluation
     */
    public final int evaluateBoard() {
        if (draw) return 0;

        int[][] groupsMap = getGroupsMap();

        int result = 0;

        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                result += groupsMap[i][j] * evaluateTile(i, j);
            }
        }


        return result;
    }

    /**
     * Calculates bonuses for the groups of the pieces based on the size of the group. Used in the evaluation function.
     *
     * @param length
     * @return bonus
     */
    protected final int mapGroupLengthToBonus(int length) {
        switch (length) {
            case 1:
                return 1;
            case 2:
                return 10;
            case 3:
                return 20;
            case 4:
                return 1000;
            default:
                throw new IllegalArgumentException("Illegal Group Length:" + length);
        }
    }


    /**
     * Creates a map of the players. For each tile of the board the method calculates which
     * player have the visible piece.
     *
     * @return map of the players
     */
    protected final int[][] getPlayerMap() {
        int[][] result = new int[BOARD_SIZE][BOARD_SIZE];
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                int w_tile = tiles[i][j] & WHITE_MASK >> WHITE_PLAYER;
                int b_tile = (tiles[i][j] & BLACK_MASK) >> BLACK;
                if ((w_tile == 0) && (b_tile == 0)) result[i][j] = DUMMY;
                else result[i][j] = (w_tile > b_tile) ? WHITE_PLAYER : BLACK;
            }
        }
        return result;
    }

    /**
     * Creates a map of the players. For each tile of the board the method calculates
     * the size of the bonus based on the number and size of the groups that the tiles
     * belongs to.
     *
     * @return map of the players
     */
    protected final int[][] getGroupsMap() {
        int[][] groupsMap = new int[BOARD_SIZE][BOARD_SIZE];
        int[][] rowGroupsMap = new int[BOARD_SIZE][BOARD_SIZE];
        int[][] colGroupsMap = new int[BOARD_SIZE][BOARD_SIZE];
        int[][] diagGroupsMap = new int[BOARD_SIZE][BOARD_SIZE];
        int[][] playerMap = getPlayerMap();

        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (playerMap[i][j] == DUMMY) continue;

                // checks the groups in rows
                if (rowGroupsMap[i][j] == 0) {
                    int size = 0;
                    for (int l = j; l < BOARD_SIZE; l++) {
                        if ((playerMap[i][j] == playerMap[i][l])) {
                            size++;
                        } else break;
                    }
                    int bonus = mapGroupLengthToBonus(size);
                    for (int l = j; l < j + size; l++) {
                        rowGroupsMap[i][l] += bonus;
                    }
                }


                // checks the groups in columns
                if (colGroupsMap[i][j] == 0) {
                    int size = 0;
                    for (int l = i; l < BOARD_SIZE; l++) {
                        if ((playerMap[i][j] == playerMap[l][j])) {
                            size++;
                        } else break;
                    }
                    int bonus = mapGroupLengthToBonus(size);
                    for (int l = i; l < i + size; l++) {
                        colGroupsMap[l][j] += bonus;
                    }
                }

                // checks the groups for the diagonals
                if ((diagGroupsMap[i][j] == 0) && (((i - j) == 0) || ((i + j) == BOARD_SIZE - 1))) {
                    int size = 0;
                    if ((i - j) == 0) {
                        for (int l = i; l < BOARD_SIZE; l++) {
                            if ((playerMap[i][j] == playerMap[l][l])) {
                                size++;
                            } else break;
                        }
                        int bonus = mapGroupLengthToBonus(size);
                        for (int l = i; l < i + size; l++) {
                            diagGroupsMap[l][l] += bonus;
                        }
                    } else { // (i+j) == BOARD_SIZE-1)
                        for (int l = i; l < BOARD_SIZE; l++) {
                            if ((playerMap[i][j] == playerMap[l][BOARD_SIZE - 1 - l])) {
                                size++;
                            } else break;
                        }
                        int bonus = mapGroupLengthToBonus(size);
                        for (int l = i; l < i + size; l++) {
                            diagGroupsMap[l][BOARD_SIZE - 1 - l] += bonus;
                        }
                    }

                }

                //sums together
                groupsMap[i][j] = rowGroupsMap[i][j] + colGroupsMap[i][j] + diagGroupsMap[i][j];
            }
        }


        return groupsMap;
    }

    protected int getPlaceablePieces(int x, int y) {
        int result = 0;
        for (int i = PIECES_COUNT - 1; i >= 0; i--) {
            if (!isPieceOn(x, y, PIECES[i])) result |= PIECES[i];
            else break;
        }
        return result;
    }

    public boolean isPiecePlaceable(int x, int y, int piece, int player) {

        int highest_w_piece = getPieceOnTop(x, y, WHITE_PLAYER);
        int highest_b_piece = getPieceOnTop(x, y, BLACK);

        if ((piece > highest_w_piece) && (piece > highest_b_piece)) return true;
        else return false;
    }

    public int getPieceOnTop(int x, int y, int player) {
        int mask = (player == WHITE_PLAYER) ? WHITE_MASK : BLACK_MASK;
        int player_tile = (tiles[x][y] & mask) >> player;
        if (player_tile > 0) {
            int piece_index = new Double(Math.floor(Math.log(player_tile) / Math.log(2))).intValue();
            return 1 << piece_index;
        } else
            return 0;
    }

    public int getMovablePiece(int x, int y, int player) {
        int highest_w_piece = getPieceOnTop(x, y, WHITE_PLAYER);
        int highest_b_piece = getPieceOnTop(x, y, BLACK);

        if (player == WHITE_PLAYER) {
            if (highest_w_piece > highest_b_piece) {
                return highest_w_piece;
            } else {
                return 0;
            }
        } else {
            if (highest_b_piece > highest_w_piece) {
                return highest_b_piece;
            } else {
                return 0;
            }
        }
    }

    public boolean isPieceOn(int x, int y, int piece, int player) {
        if ((tiles[x][y] & (piece << player)) > 0)
            return true;
        else
            return false;
    }

    public boolean isPieceOn(int x, int y, int piece) {
        return isPieceOn(x, y, piece, WHITE_PLAYER) || isPieceOn(x, y, piece, BLACK);
    }

    /**
     * Method calculates whether the board represent a terminal state of the game
     *
     * @param playerToMove
     * @return
     */
    public int isTerminate(int playerToMove) {
        int win_player = DUMMY;
        if (draw) return DRAW;
        int[][] groupsMap = getGroupsMap();
        int[][] playerMap = getPlayerMap();
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if ((win_player != playerToMove) && (groupsMap[i][j] >= 1000)) {
                    win_player = playerMap[i][j];
                }
            }
        }
        return win_player;
    }

    public long calculateSimpleHash() {
        long hash = 0;
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                hash = 31 * hash + tiles[i][j];
            }
        }
        return hash;
    }

    @Override
    public int hashCode() {
        return (int) calculateSimpleHash();
    }

    @Override
    public boolean equals(Object obj) {
        Board o = (Board) obj;
        if (this.hashCode() != o.hashCode())
            return false;
        for (int i = 0; i < BOARD_SIZE + 1; i++) {
            for (int j = 0; j < BOARD_SIZE + 1; j++) {
                if (tiles[i][j] != o.tiles[i][j]) return false;
            }
        }
        return true;
    }

    public int[][] getTiles() {
        return tiles;
    }

}
