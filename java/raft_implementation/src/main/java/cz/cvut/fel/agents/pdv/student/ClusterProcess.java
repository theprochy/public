package cz.cvut.fel.agents.pdv.student;

import cz.cvut.fel.agents.pdv.dsand.Message;
import cz.cvut.fel.agents.pdv.dsand.Pair;
import cz.cvut.fel.agents.pdv.raft.RaftProcess;
import cz.cvut.fel.agents.pdv.raft.messages.*;

import java.util.*;
import java.util.function.BiConsumer;

/**
 * Vasim ukolem bude naimplementovat (pravdepodobne nejenom) tuto tridu. Procesy v clusteru pracuji
 * s logy, kde kazdy zanam ma podobu mapy - kazdy zaznam v logu by mel reprezentovat stav
 * distribuovane databaze v danem okamziku.
 *
 * Vasi implementaci budeme testovat v ruznych scenarich (viz evaluation.RaftRun a oficialni
 * zadani). Nasim cilem je, abyste zvladli implementovat jednoduchou distribuovanou key/value
 * databazi s garancemi podle RAFT.
 */

public class ClusterProcess extends RaftProcess<Map<String, String>> {

    private static final Random random = new Random();

    public enum State {
        FOLLOWER,
        WANTS_TO_LEAD,
        LEADER
    }

    private int epoch;
    private int timeout;
    private int commitIndex;
    private int voteRemaining;
    private final int networkDelays;
    private boolean voted;
    private State state;
    private String leaderId;

    private final List<String> otherProcessesInCluster;
    private final List<Entry> log = new ArrayList<>();
    private final Set<String> requestsReceived = new HashSet<>();
    private final Map<String, String> database = new HashMap<>();
    private final Map<String, Integer> prevLogIndexByProcess = new HashMap<>();
    private final Map<Integer, ClientRequestWithContent> waitingForResponse = new HashMap<>();

    public ClusterProcess(String id, Queue<Message> inbox, BiConsumer<String, Message> outbox,
                          List<String> otherProcessesInCluster, int networkDelays) {
        super(id, inbox, outbox);
        this.otherProcessesInCluster = otherProcessesInCluster;
        this.networkDelays = networkDelays;

        this.commitIndex = -1;
        this.voted = false;
        this.state = State.FOLLOWER;
        this.timeout = getTimeout();
        this.leaderId = null;
    }

    @Override
    public Optional<Map<String, String>> getLastSnapshotOfLog() {
        if (database.isEmpty()) {
            return Optional.empty();
        } else {
            return Optional.of(database);
        }
    }

    @Override
    public String getCurrentLeader() {
        return leaderId;
    }

    @Override
    public void act() {
        while (!inbox.isEmpty()) {

            Message m = inbox.poll();

            if (m instanceof ClientRequest) {
                System.out.println(m.getClass());
                if (m instanceof ClientRequestWhoIsLeader ||
                    m instanceof ClientRequestWithContent && state != State.LEADER) {
                    String reqId = ((ClientRequest) m).getRequestId();
                    send(m.sender, new ServerResponseLeader(reqId, leaderId));
                }

                if (m instanceof ClientRequestWithContent && state == State.LEADER) {
                    ClientRequestWithContent req = (ClientRequestWithContent) m;
                    if (!requestsReceived.contains(req.getRequestId())) {
                        requestsReceived.add(req.getRequestId());

                        String name = req.getOperation().getName();
                        String first = ((Pair<String, String>) (req.getContent())).getFirst();
                        String second = ((Pair<String, String>) (req.getContent())).getSecond();
                        if (req.getOperation().getName().equals("GET")) {
                            send(m.sender, new ServerResponseWithContent<>(
                                                req.getRequestId(),
                                                database.get(first)));
                        } else {
                            waitingForResponse.put(log.size(), req);
                            log.add(new Entry(epoch, name, first, second));
                        }
                    }
                }
            }

            if (m instanceof EpochMessage) {
                int msgEpoch = ((EpochMessage) m).epoch;

                if (msgEpoch < epoch) {
                    continue;
                } else if (msgEpoch > epoch) {
                    voted = false;
                    epoch = msgEpoch;
                    state = State.FOLLOWER;
                    timeout = getTimeout();
                }
            }

            if (m instanceof EntriesAppendedMessage) {
                int prevLogIndex = ((EntriesAppendedMessage) m).getLogSize();
                prevLogIndexByProcess.put(m.sender, prevLogIndex);
            } else if (m instanceof AppendEntriesMessage) {
                state = State.FOLLOWER;
                timeout = getTimeout();
                leaderId = m.sender;
                AppendEntriesMessage msg = (AppendEntriesMessage) m;

                while (log.size() - 1 > msg.getPrevLogIndex()) {
                    log.remove(log.size() - 1);
                }
                log.addAll(msg.getEntries());

                if (msg.getCommitIndex() > commitIndex) {
                    int oldIndex = commitIndex;
                    commitIndex = Math.min(log.size() - 1, msg.getCommitIndex());

                    while (oldIndex < commitIndex) {
                        oldIndex++;
                        performOperation(database, log.get(oldIndex));
                    }
                }

                send(m.sender, new EntriesAppendedMessage(epoch, log.size() - 1));

            } else if (m instanceof RequestVoteMessage && state == State.FOLLOWER) {
                if (voted) continue;
                voted = true;
                timeout = getTimeout();
                send(m.sender, new VoteGivenMessage(epoch));

            } else if (m instanceof VoteGivenMessage && state == State.WANTS_TO_LEAD) {
                voteRemaining--;
                if (voteRemaining == 0) {
                    state = State.LEADER;
                    leaderId = getId();
                    int decrement = commitIndex == -1 ? 0 : 1;
                    for (String id : otherProcessesInCluster) {
                        prevLogIndexByProcess.put(id, commitIndex - decrement);
                    }
                }
            }
        }

        if (state == State.LEADER) {
            boolean cont = true;
            while (cont && commitIndex < log.size()) {
                cont = false;
                int quorum = otherProcessesInCluster.size() / 2 + 1;
                for (Map.Entry<String, Integer> entry : prevLogIndexByProcess.entrySet()) {
                    if (entry.getValue() > commitIndex) { quorum--;}
                    if (quorum == 0 && log.get(commitIndex + 1).getEpoch() == epoch) {
                        commitIndex++;
                        performOperation(database, log.get(commitIndex));
                        String sender = waitingForResponse.get(commitIndex).sender;
                        String reqId = waitingForResponse.get(commitIndex).getRequestId();
                        System.out.println("size " + waitingForResponse.size());
                        System.out.println(commitIndex);
                        send(sender, new ServerResponseConfirm(reqId));
                        cont = true;
                        break;
                    }
                }
            }

            for (String id : otherProcessesInCluster) {
                int from = prevLogIndexByProcess.get(id);
                send(id, new AppendEntriesMessage(epoch,
                        from,
                        from == -1 ? 0 : log.get(from).getEpoch(),
                        commitIndex,
                        new ArrayList<>(log.subList(from + 1, log.size()))));
            }
        } else {
            if (timeout == 0) {
                startElection();
            } else {
                timeout--;
            }
        }
    }

    private void startElection() {
        if (state == State.FOLLOWER || state == State.WANTS_TO_LEAD) {
            epoch++;
            state = State.WANTS_TO_LEAD;
            voteRemaining = otherProcessesInCluster.size() / 2 + 1;
            timeout = getTimeout();

            for (String id : otherProcessesInCluster) {
                if (log.size() == 0) {
                    send(id, new RequestVoteMessage(epoch, 0, -1));
                } else {
                    send(id, new RequestVoteMessage(epoch, log.size(), log.get(log.size() - 1).getEpoch()));
                }
            }
        }
    }

    private int getTimeout() {
        return 2 * networkDelays + 2 + random.nextInt(2 * otherProcessesInCluster.size());
    }

    private void performOperation(Map<String, String> database, Entry entry) {
        if (entry.getOperation().equals("PUT")) {
            database.put(entry.getKey(), entry.getValue());
        } else if (entry.getOperation().equals("APPEND")) {
            if (database.containsKey(entry.getKey())) {
                database.put(entry.getKey(), database.get(entry.getKey()) + entry.getValue());
            } else {
                database.put(entry.getKey(), entry.getValue());
            }
        }
    }
}
