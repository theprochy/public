#include "iddfs.h"
#include <climits>
#include <unordered_map>

using namespace std;

// Naimplementujte efektivni algoritmus pro nalezeni nejkratsi (respektive nej-
// levnejsi) cesty v grafu. V teto metode mate ze ukol naimplementovat pametove
// efektivni algoritmus pro prohledavani velkeho stavoveho prostoru. Pocitejte
// s tim, ze Vami navrzeny algoritmus muze bezet na stroji s omezenym mnozstvim
// pameti (radove nizke stovky megabytu). Vhodnym pristupem tak muze byt napr.
// iterative-deepening depth-first search.
//
// Metoda ma za ukol vratit ukazatel na cilovy stav, ktery je dosazitelny pomoci
// nejkratsi/nejlevnejsi cesty. Pokud je nejkratsich cest vice, vratte ukazatel
// na stav s nejnizsim identifikatorem (viz methoda 'state::get_identifier()').

shared_ptr <const state> goal = nullptr;
unsigned long long int goal_id = ULLONG_MAX;
unordered_map<unsigned long long int, int> depths;

#pragma GCC optimize("Ofast,unroll-loops,fast-math")
void iddfs_h(const shared_ptr<const state> &root, int depth, int max_depth) {

    for (shared_ptr<const state> cur : root->next_states()) {

        if (depth + 1 >= max_depth) {
            if (cur->is_goal()) {
                if (cur->get_identifier() < goal_id) {
                    #pragma omp critical
                    {
                        goal = cur;
                        goal_id = cur->get_identifier();
                    }
                }
            }
            continue;
        }

        if (max_depth - depth < 3) {
            iddfs_h(cur, depth + 1, max_depth);
            continue;
        }

        if (!depths.count(cur->get_identifier())) {
            #pragma omp critical
            {
                depths[cur->get_identifier()] = depth + 1;
            }

            #pragma omp task
            iddfs_h(cur, depth + 1, max_depth);
        } else {

            if (depth + 1 < depths.at(cur->get_identifier())) {
                #pragma omp critical
                {
                    depths[cur->get_identifier()] = depth + 1;
                }

                #pragma omp task
                iddfs_h(cur, depth + 1, max_depth);
            } else if (depth + 1 == depths.at(cur->get_identifier())) {
                #pragma omp task
                iddfs_h(cur, depth + 1, max_depth);
            }
        }
    }
}


std::shared_ptr<const state> iddfs(std::shared_ptr<const state> root) {

    if(root->is_goal()) return root;
    depths[root->get_identifier()] = 0;

    int i = 0;

    while (!goal) {
        #pragma omp parallel
        {
            #pragma omp single
            iddfs_h(root, 0, i++);
        }
    }

    return goal;
}
