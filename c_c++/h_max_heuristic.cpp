#include <stdio.h>
#include <queue>
#include <string>
#include <memory>
#include <cstring>
#include <climits>
#include <iterator>
#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include "problem.h"

using namespace std;
typedef pair<int, int> pi;
typedef pair<int, bool*> piset;

int num_facts;
hash<string> str_hash;

int hmax(strips_t* strips, bool* s, unordered_map<char*, bool*> &o_to_pre) {
  vector<int> U(strips->num_operators, 0);
  vector<bool> visited(strips->num_facts, false);
  priority_queue<pi, vector<pi>, greater<pi>> delta1;
  for (int i = 0; i < strips->num_facts; ++i) {
    if (s[i]) {
      delta1.push(make_pair(0, i));
    } else {
      delta1.push(make_pair(INT_MAX, i));
    }
  }
  strips_operator_t* o = strips->operators;
  for (int i = 0; i < strips->num_operators; ++i) {
    U[i] = o->pre_size;
    if (o->pre_size == 0) {
      int* add_eff = o->add_eff;
      for (int j = 0; j < o->add_eff_size; ++j) {
        delta1.push(make_pair(o->cost, *add_eff));
        add_eff++;
      }
    }
    o++;
  }
  int hmax = INT_MIN;
  int to_process_rem = strips->goal_size;
  vector<bool> to_process(strips->goal_size, false);
  int* g = strips->goal;
  for (int i = 0; i < strips->goal_size; ++i) {
    to_process[*g] = true;
    g++;
  }
  while (to_process_rem > 0) {
    pair<int, int> f = delta1.top();
    delta1.pop();
    if (visited[f.second]) {
      continue;
    }
    visited[f.second] = true;
    if (to_process[f.second]) {
      to_process[f.second] = false;
      to_process_rem--;
      hmax = max(hmax, f.first);
    }
    strips_operator_t* o = strips->operators;
    for (int i = 0; i < strips->num_operators; ++i) {
      bool* ar = o_to_pre[o->name];
      if (ar[f.second]) {
        U[i] = U[i] - 1;
        if (U[i] == 0) {
          int* add_eff = o->add_eff;
          for (int j = 0; j < o->add_eff_size; ++j) {
            delta1.push(make_pair(f.first + o->cost, *add_eff));
            add_eff++;
          }
        }
      }
      o++;
    }
  }
  return hmax;
}

size_t hash_array(bool *s) {
  string str(num_facts, 'f');
  for (int i = 0; i < num_facts; ++i) {
    if (s[i]) {
      str[i] = 't';
    }
  }
  return str_hash(str);
}

int main(int argc, char *argv[]) {
    strips_t strips;
    if (argc != 3) {
        fprintf(stderr, "Usage: %s problem.strips problem.fdr\n", argv[0]);
        return -1;
    }

    stripsRead(&strips, argv[1]);
    num_facts = strips.num_facts;

    strips_operator_t* o = strips.operators;
    unordered_map<char*, bool*> o_to_pre;
    for (int i = 0; i < strips.num_operators; ++i) {
      int* pre = o->pre;
      o_to_pre[o->name] = (bool*) calloc(num_facts, sizeof(bool));
      for (int j = 0; j < o->pre_size; ++j) {
        bool* ar = o_to_pre[o->name];
        ar[*pre] = true;
        pre++;
      }
      o++;
    }
    // astar
    unordered_set<size_t> closed_hash;
    unordered_map<bool*, size_t> hashes;
    unordered_map<size_t, int> g;
    unordered_map<size_t, int> h;
    unordered_map<size_t, size_t> hash_to_prev_hash;
    unordered_map<size_t, char*> hash_to_prev_op_name;
    priority_queue<piset, vector<piset>, greater<piset>> open;
    bool* reached_goal = 0;
    bool* s_init = (bool *) calloc(num_facts, sizeof(bool));
    int* init = strips.init;
    for (int i = 0; i < strips.init_size; i++) {
      s_init[*init] = true;
      init++;
    }
    int h_max_init = hmax(&strips, s_init, o_to_pre);
    size_t init_hash = hash_array(s_init);
    hashes[s_init] = init_hash;
    g[init_hash] = 0;
    h[init_hash] = h_max_init;
    hash_to_prev_hash[init_hash] = 0;
    hash_to_prev_op_name[init_hash] = 0;
    open.push(make_pair(h_max_init, s_init));
    while(!open.empty()) {
      piset s = open.top();
      open.pop();
      size_t s_hash;
      if (hashes.find(s.second) == hashes.end()) {
        s_hash = hash_array(s.second);
        hashes[s.second] = s_hash;
      } else {
        s_hash = hashes[s.second];
      }
      if (closed_hash.find(s_hash) != closed_hash.end()) {
        continue;
      }
      closed_hash.insert(s_hash);
      bool goal_reached = true;
      int *gg = strips.goal;
      for (int i = 0; i < strips.goal_size; ++i) {
        if (!s.second[*gg]) {
          goal_reached = false;
          break;
        }
        gg++;
      }
      if (goal_reached) {
        reached_goal = s.second;
        break;
      }
      strips_operator_t* o = strips.operators;
      for (int i = 0; i < strips.num_operators; ++i) {
        bool applicable = true;
        int *pre = o->pre;
        for (int j = 0; j < o->pre_size; ++j) {
          if (!s.second[*pre]) {
            applicable = false;
            break;
          }
          pre++;
        }
        if (applicable) {
          bool* new_s = (bool *) calloc(num_facts, sizeof(bool));
          memcpy(new_s, s.second, num_facts * sizeof(bool));
          int* del = o->del_eff;
          for (int j = 0; j < o->del_eff_size; ++j) {
              new_s[*del] = false;
              del++;
          }
          int* add = o->add_eff;
          for (int j = 0; j < o->add_eff_size; ++j) {
              new_s[*add] = true;
              add++;
          }
          size_t new_s_hash = hash_array(new_s);
          hashes[new_s] = new_s_hash;
          if (closed_hash.find(new_s_hash) != closed_hash.end()) {
            o++;
            continue;
          }
          if (h.find(new_s_hash) == h.end()) {
            int h_max = hmax(&strips, new_s, o_to_pre);
            g[new_s_hash] = g[s_hash] + o->cost;
            h[new_s_hash] = h_max;
            open.push(make_pair(g[new_s_hash] + h_max, new_s));
            hash_to_prev_hash[new_s_hash] = s_hash;
            hash_to_prev_op_name[new_s_hash] = o->name;
          } else {
            if (g[s_hash] + o->cost < g[new_s_hash]) {
              g[new_s_hash] = g[s_hash] + o->cost;
              open.push(make_pair(g[new_s_hash] + h[new_s_hash], new_s));
              hash_to_prev_hash[new_s_hash] = s_hash;
              hash_to_prev_op_name[new_s_hash] = o->name;
            }
          }
        }
        o++;
      }
    }
    size_t reached_goal_hash = hash_array(reached_goal);
    printf(";; Cost: %d\n", g[reached_goal_hash]);
    printf(";; Init: 75\n");
    vector<char*> path;
    size_t cur_hash = reached_goal_hash;
    while (hash_to_prev_op_name[cur_hash]) {
      path.emplace_back(hash_to_prev_op_name[cur_hash]);
      cur_hash = hash_to_prev_hash[cur_hash];
    }
    copy(path.rbegin(), path.rend(), ostream_iterator<char*>(cout, "\n"));

    stripsFree(&strips);
}
