package cz.cvut.fel.agents.pdv.student;

import cz.cvut.fel.agents.pdv.dsand.Message;

public abstract class EpochMessage extends Message {

    public final int epoch;

    public EpochMessage(int epoch) {
        this.epoch = epoch;
    }
}
