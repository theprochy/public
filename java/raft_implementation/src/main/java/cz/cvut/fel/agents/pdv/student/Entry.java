package cz.cvut.fel.agents.pdv.student;

import java.io.Serializable;

public class Entry implements Serializable {

    private int epoch;
    private String operation;
    private String key;
    private String value;

    public Entry(int epoch, String operation, String key, String value) {
        this.operation = operation;
        this.epoch = epoch;
        this.key = key;
        this.value = value;
    }

    public int getEpoch() {
        return epoch;
    }

    public String getOperation() {
        return operation;
    }

    public String getKey() {
        return key;
    }

    public String getValue() {
        return value;
    }
}
