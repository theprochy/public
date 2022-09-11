package cz.cvut.fel.agents.pdv.student;

import java.util.List;

public class AppendEntriesMessage extends EpochMessage {

    private int prevLogIndex;
    private int prevLogTerm;
    private int commitIndex;
    private List<Entry> entries;

    public AppendEntriesMessage(int epoch, int prevLogIndex, int prevLogTerm,
                                int commitIndex, List<Entry> entries) {
        super(epoch);
        this.prevLogIndex = prevLogIndex;
        this.prevLogTerm = prevLogTerm;
        this.commitIndex = commitIndex;
        this.entries = entries;
    }

    public int getPrevLogIndex() {
        return prevLogIndex;
    }

    public int getPrevLogTerm() {
        return prevLogTerm;
    }

    public int getCommitIndex() {
        return commitIndex;
    }

    public List<Entry> getEntries() {
        return entries;
    }
}
