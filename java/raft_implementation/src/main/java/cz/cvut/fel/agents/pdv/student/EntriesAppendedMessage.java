package cz.cvut.fel.agents.pdv.student;

public class EntriesAppendedMessage extends EpochMessage {

    private int logSize;

    public EntriesAppendedMessage(int epoch, int logSize) {
        super(epoch);
        this.logSize = logSize;
    }

    public int getLogSize() {
        return logSize;
    }
}
