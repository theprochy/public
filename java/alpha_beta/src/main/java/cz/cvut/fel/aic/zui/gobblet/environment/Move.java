package cz.cvut.fel.aic.zui.gobblet.environment;

/**
 * Encapsulation of data defining a move.
 *
 * @author BB
 */
public class Move {
    private int from_x, from_y, to_x, to_y;
    private int piece;


    public Move(int fromX, int fromY, int toX, int toY, int piece, int player) {
        from_x = fromX;
        from_y = fromY;
        to_x = toX;
        to_y = toY;
        this.piece = piece;
        this.player = player;
    }

    public int getPiece() {
        return piece;
    }

    public void setPiece(int piece) {
        this.piece = piece;
    }

    public int getPlayer() {
        return player;
    }

    public void setPlayer(int player) {
        this.player = player;
    }

    private int player;

    public int getFrom_x() {
        return from_x;
    }

    public void setFrom_x(int fromX) {
        from_x = fromX;
    }

    public int getFrom_y() {
        return from_y;
    }

    public void setFrom_y(int fromY) {
        from_y = fromY;
    }

    public int getTo_x() {
        return to_x;
    }

    public void setTo_x(int toX) {
        to_x = toX;
    }

    public int getTo_y() {
        return to_y;
    }

    public void setTo_y(int toY) {
        to_y = toY;
    }

    @Override
    public String toString() {
        return "Piece: " + piece + ", Player: " + player + ", [" + ((char) (from_x + 65)) + "," + from_y + "] -> " + "[" + ((char) (to_x + 65)) + "," + to_y + "]";
    }
}
