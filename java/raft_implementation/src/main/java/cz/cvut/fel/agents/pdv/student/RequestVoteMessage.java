package cz.cvut.fel.agents.pdv.student;

public class RequestVoteMessage extends EpochMessage {

    private int logSize;
    private int lastEntryEpoch;

    public RequestVoteMessage(int epoch, int logSize, int lastEntryEpoch) {
        super(epoch);
        this.logSize = logSize;
        this.lastEntryEpoch = lastEntryEpoch;
    }

    public int getLogSize() {
        return logSize;
    }

    public int getLastEntryEpoch() {
        return lastEntryEpoch;
    }
}
