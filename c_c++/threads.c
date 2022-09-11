#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define _PHASE_COUNT 6

enum place {
    NUZKY,
    VRTACKA,
    OHYBACKA,
    SVARECKA,
    LAKOVNA,
    SROUBOVAK,
    FREZA,
    _PLACE_COUNT
};

enum product { A, B, C, _PRODUCT_COUNT };

typedef struct worker {
    int workplace;
    char *name;
    char working;
    char tried_working;
    char go_home;
    char went_home;
    pthread_t thread;
} worker;

typedef struct node_l {
    worker *data;
    struct node_l *next;
} node_l;

int shift_over = 0;
int ready_places[_PLACE_COUNT] = {0};
int parts[_PRODUCT_COUNT][_PHASE_COUNT] = {0};
int durations[_PLACE_COUNT] = {100, 200, 150, 300, 400, 250, 500};

node_l *root;
pthread_cond_t state_changed = PTHREAD_COND_INITIALIZER;
pthread_mutex_t state_mx = PTHREAD_MUTEX_INITIALIZER;

const char *place_str[_PLACE_COUNT] = {
        [NUZKY] = "nuzky",       [VRTACKA] = "vrtacka",
        [OHYBACKA] = "ohybacka", [SVARECKA] = "svarecka",
        [LAKOVNA] = "lakovna",   [SROUBOVAK] = "sroubovak",
        [FREZA] = "freza",
};

const char *product_str[_PRODUCT_COUNT] = {
        [A] = "A", [B] = "B", [C] = "C",
};

const int work_order[_PRODUCT_COUNT][_PLACE_COUNT] = {
        [A] = {NUZKY, VRTACKA, OHYBACKA, SVARECKA, VRTACKA, LAKOVNA},
        [B] = {VRTACKA, NUZKY, FREZA, VRTACKA, LAKOVNA, SROUBOVAK},
        [C] = {FREZA, VRTACKA, SROUBOVAK, VRTACKA, FREZA, LAKOVNA},
};

int find_string_in_array(const char **array, int length, char *what) {
    for (int i = 0; i < length; i++)
        if (strcmp(array[i], what) == 0)
            return i;
    return -1;
}

int can_work(worker *w) {
    for (int i = _PHASE_COUNT - 1; i >= 0; --i) {
        for (int j = 0; j < _PRODUCT_COUNT; ++j) {
            if (work_order[j][i] == w->workplace &&
            	ready_places[w->workplace] > 0 &&
                parts[j][i] > 0) {
                	return 1;
            }
        }
    }
    return 0;
}

void *worker_th(void *workr) {
    worker *w = (worker *)workr;
    int wp = w->workplace;

    pthread_mutex_lock(&state_mx);
    while (1) {

        if (w->go_home) {
            w->went_home = 1;
            pthread_mutex_unlock(&state_mx);
            return 0;
        }

        char worked = 0;
        for (int i = _PHASE_COUNT - 1; i >= 0; --i) {
            for (int j = 0; j < _PRODUCT_COUNT; ++j) {
                if (work_order[j][i] == wp && ready_places[wp] > 0 &&
                    parts[j][i] > 0) {
                    worked = 1;
                    w->working = 1;
                    ready_places[w->workplace]--;
                    parts[j][i]--;

                    printf("%s %s %d %s\n", w->name, place_str[w->workplace],
                           i + 1, product_str[j]);
                    if (i == _PHASE_COUNT - 1) {
                        printf("done %s\n", product_str[j]);
                    }
                    pthread_mutex_unlock(&state_mx);

                    usleep(durations[wp] * 1000);

                    pthread_mutex_lock(&state_mx);
                    if (i != _PHASE_COUNT - 1) {
                        parts[j][i + 1]++;
                    }
                    pthread_cond_broadcast(&state_changed);
                    w->working = 0;
                    ready_places[w->workplace]++;
                }
            }
        }

//        fprintf(stderr, "n: %s so: %d wkd: %d\n", w->name, shift_over, worked);
        if (!worked && shift_over) {
            node_l *cur = root;
            char working = 0;
            while (cur->next) {
                if (cur->data == w || cur->data->went_home) {
                    cur = cur->next;
                    continue;
                }
                working |= cur->data->working;
                working |= can_work(cur->data);
                cur = cur->next;
            }
            if (!working) {
//                fprintf(stderr, "####%s %s going home\n", w->name,
//                        place_str[wp]);
                w->went_home = 1;
                pthread_cond_broadcast(&state_changed);
                pthread_mutex_unlock(&state_mx);
                return 0;
            }
        }

        if (!worked) {
            pthread_cond_wait(&state_changed, &state_mx);
        }
    }
}

int main(int argc, char **argv) {

    root = malloc(sizeof(node_l));
    root->next = NULL;
    node_l *end = root;

    while (1) {
        char *line, *cmd, *arg1, *arg2, *arg3, *saveptr = NULL;
        int s = scanf(" %m[^\n]", &line);
        if (s == EOF)
            break;
        if (s == 0)
            continue;

        fprintf(stderr, "%s\n", line);

        cmd = strtok_r(line, " ", &saveptr);
        arg1 = strtok_r(NULL, " ", &saveptr);
        arg2 = strtok_r(NULL, " ", &saveptr);
        arg3 = strtok_r(NULL, " ", &saveptr);

        if (strcmp(cmd, "start") == 0 && arg1 && arg2 && !arg3) {
            worker *w = malloc(sizeof(worker));
            w->name = strdup(arg1);
            w->working = 0;
            w->tried_working = 0;
            w->went_home = 0;
            w->go_home = 0;
            w->workplace = find_string_in_array(place_str, _PLACE_COUNT, arg2);

            end->data = w;
            end->next = malloc(sizeof(node_l));
            end = end->next;
            end->next = NULL;

            pthread_create(&(w->thread), NULL, &worker_th, (void *)w);

        } else if (strcmp(cmd, "make") == 0 && arg1 && !arg2) {
            int product =
                find_string_in_array(product_str, _PRODUCT_COUNT, arg1);
            if (product >= 0) {
                pthread_mutex_lock(&state_mx);

                parts[product][0]++;
                pthread_cond_broadcast(&state_changed);
                pthread_mutex_unlock(&state_mx);
            }
        } else if (strcmp(cmd, "end") == 0 && arg1 && !arg2) {
            node_l *cur = root;
            while (cur->next) {
                if (strcmp(cur->data->name, arg1) == 0) {
                    cur->data->go_home = 1;
                    break;
                }
                cur = cur->next;
            }
        } else if (strcmp(cmd, "add") == 0 && arg1 && !arg2) {
            int wp = find_string_in_array(place_str, _PLACE_COUNT, arg1);

            pthread_mutex_lock(&state_mx);
            ready_places[wp]++;
            pthread_mutex_unlock(&state_mx);

            pthread_cond_broadcast(&state_changed);

        } else if (strcmp(cmd, "remove") == 0 && arg1 && !arg2) {
            int wp = find_string_in_array(place_str, _PLACE_COUNT, arg1);
            pthread_mutex_lock(&state_mx);
            ready_places[wp]--;

            pthread_mutex_unlock(&state_mx);

        } else {
            fprintf(stderr, "Invalid command: %s\n", line);
        }
        free(line);
    }

    shift_over = 1;
    pthread_cond_broadcast(&state_changed);

    node_l *cur = root;

    while (cur->next) {
        pthread_join(cur->data->thread, NULL);
        cur = cur->next;
    }

    while (root->next) {
        node_l *temp = root;
        root = root->next;
        free(temp->data->name);
        free(temp->data);
        free(temp);
    }

    pthread_cond_destroy(&state_changed);

    return 0;
}
