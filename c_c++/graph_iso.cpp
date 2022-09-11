#include <stack>
#include <vector>
#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include <bits/stdc++.h>

using namespace std;

int *depths;
unordered_map<int, long> hashByNode;
vector<vector<vector<int>>> graphs;

long hashGraph(vector<vector<int>> &graph, int v, stack<int> &s, unordered_set<int> &sSet);
void hashFromNode(vector<vector<int>> &graph, int v, int depth, stack<int> &s, unordered_set<int> &sSet);
bool findRoot(vector<vector<int>> &graph, int v, unordered_set<int> &closed, stack<int> &s, unordered_set<int> &sSet);

int main() {
    int t, n, m;
    cin >> t >> n >> m;
    depths = new int[n];
    for (int i = 0; i < t; i++) {
        vector<vector<int>> graph(n, vector<int>());
        for (int j = 0; j < m; j++) {
            int u, v;
            cin >> u >> v;
            graph[--u].push_back(--v);
            graph[v].push_back(u);
        }
        graphs.push_back(graph);
    }
    unordered_map<long, int> hashCounts;
    for (auto graph : graphs) {
        for (int i = 0; i < n; i++) {
            if (graph[i].size() > 1) {
                stack<int> s, t;
                unordered_set<int> a,b,c;
                findRoot(graph, i, a, s, b);
                int hash = hashGraph(graph, s.top(), t, c);
                if (hashCounts.find(hash) == hashCounts.end()) {
                  hashCounts[hash] = 1;
                } else {
                  hashCounts[hash]++;
                }
                hashByNode.clear();
                break;
            }
        }
    }
    vector<int> counts;
    for (const auto &e : hashCounts) {
      counts.push_back(e.second);
    }
    sort(counts.begin(), counts.end());
    for (auto i : counts) {
      cout << i << " ";
    }
    cout << endl;
}

long hashGraph(vector<vector<int>> &graph, int v, stack<int> &s, unordered_set<int> &sSet) {
    s.push(v);
    sSet.insert(v);
    long hash = 0;
    for (int w : graph[v]) {
        hashFromNode(graph, w, 1, s, sSet);
        hash += hashByNode[v];
    }
    return hash;
}

void hashFromNode(vector<vector<int>> &graph, int v, int depth, stack<int> &s, unordered_set<int> &sSet) {
    int prev = s.empty() ? -1 : s.top();
    if (graph[v].size() == 1) {
        hashByNode[prev] = 1L;
        return;
    }
    depths[v] = depth;
    s.push(v);
    sSet.insert(v);
    for (int w : graph[v]) {
        if (sSet.find(v) == sSet.end()) {
            break;
        }
        if (w == prev || hashByNode.find(w) != hashByNode.end()) {
            continue;
        } else if (sSet.find(w) != sSet.end()) {
            int x = 0, x_prev = -1;
            int l = depth - depths[w] + 1;
            long hash = 0;
            for (int i = 1; i < l; i++) {
                int c = min(i, l - i);
                if (i > 1) {
                    x_prev = x;
                }
                x = s.top();
                if (hashByNode.find(x) != hashByNode.end()) {
                    hash += c * l * hashByNode[x];
                } else {
                    for (int next : graph[x]) {
                        if (sSet.find(next) != sSet.end() || next == x_prev) {
                            continue;
                        }
                        hashFromNode(graph, next, depth - i + 2, s, sSet);
                        hash += c * l * hashByNode[x];
                    }
                }
                s.pop();
                sSet.erase(x);
            }
            sSet.erase(s.top());
            s.pop();
            hashByNode[s.top()] = hash * l;
        } else {
            hashFromNode(graph, w, depth + 1, s, sSet);
        }
    }
}

bool findRoot(vector<vector<int>> &graph, int v, unordered_set<int> &closed, stack<int> &s, unordered_set<int> &sSet) {
    int prev = s.empty() ? -1 : s.top();
    s.push(v);
    sSet.insert(v);
    for (int w : graph[v]) {
        if (w == prev || graph[w].size() == 1 || closed.find(w) != closed.end()) {
            continue;
        } else if (sSet.find(w) != sSet.end()) {
            int x;
            do {
                x = s.top();
                for (int next : graph[x]) {
                    if (graph[next].size() == 1 || sSet.find(next) != sSet.end() || closed.find(next) != closed.end()) {
                        continue;
                    }
                    if (findRoot(graph, next, closed, s, sSet)) {
                        return true;
                    }
                }
                s.pop();
                sSet.erase(x);
                closed.insert(x);
            } while (x != w);
        } else {
            if (findRoot(graph, w, closed, s, sSet)) {
                return true;
            }
        }
    }
    if (s.top() == v) {
        return true;
    }
    return false;
}
