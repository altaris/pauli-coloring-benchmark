#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ============================================================================
// DICT
// ============================================================================

// a dict[int, list[int]], keys are {0, 1, ..., n_keys - 1}
struct dict
{
    int *values;
    size_t *start;
    size_t *length;
    size_t n_keys;
};
int *dict_get(const struct dict *d, int key, size_t idx);
struct dict *dict_invert(const struct dict *d, size_t n_keys);
size_t dict_n_vals(const struct dict *d);
struct dict *dict_new(size_t n_keys, size_t n_values);

void dict_free(struct dict *d)
{
    free((int *)d->values);
    free((int *)d->start);
    free((int *)d->length);
    free(d);
}

int *dict_get(const struct dict *d, int key, size_t idx)
{
    return d->values + d->start[key] + idx;
}

// Dict inversion, but the number of keys of the inverted dict is already known
struct dict *dict_invert(const struct dict *d, size_t n_keys)
{
    struct dict *e = dict_new(n_keys, dict_n_vals(d));
    for (size_t dk = 0; dk < d->n_keys; dk++)
    {
        for (size_t i = 0; i < d->length[dk]; i++)
        {
            e->length[*dict_get(d, dk, i)]++;
        }
    }
    int **fill_idx = calloc(n_keys, sizeof(int *));
    fill_idx[0] = e->values;
    for (size_t ek = 1; ek < n_keys; ek++)
    {
        e->start[ek] = e->start[ek - 1] + e->length[ek - 1];
        fill_idx[ek] = fill_idx[ek - 1] + e->length[ek - 1];
    }
    for (size_t dk = 0; dk < d->n_keys; dk++)
    {
        for (size_t j = 0; j < d->length[dk]; j++)
        {
            int ek = *dict_get(d, dk, j);
            *fill_idx[ek] = dk;
            fill_idx[ek]++;
        }
    }
    return e;
}

size_t dict_n_vals(const struct dict *d)
{
    size_t n_vals = 0;
    for (size_t i = 0; i < d->n_keys; i++)
    {
        n_vals += d->length[i];
    }
    return n_vals;
}

struct dict *dict_new(size_t n_keys, size_t n_values)
{
    struct dict *d = malloc(sizeof(struct dict));
    d->values = calloc(n_values, sizeof(int));
    d->start = calloc(n_keys, sizeof(size_t));
    d->length = calloc(n_keys, sizeof(size_t));
    d->n_keys = n_keys;
    return d;
}

// ============================================================================
// SET
// ============================================================================

// a set[int] of ints between 0 and a set max_val
struct set
{
    bool *has_val;
    size_t max_val;
};
bool set_add(struct set *s, int val);
void set_free(struct set *s);
bool set_in(const struct set *s, int val);
struct set *set_new(size_t max_val);
int set_smallest_not_in(const struct set *s);

bool set_add(struct set *s, int val)
{
    if (val >= s->max_val)
    {
        return false;
    }
    s->has_val[val] = true;
    return true;
}

void set_free(struct set *s)
{
    free(s->has_val);
    free(s);
}

bool set_in(const struct set *s, int val)
{
    return s->has_val[val];
}

struct set *set_new(size_t max_val)
{
    struct set *s = malloc(sizeof(struct set));
    s->has_val = calloc(max_val, sizeof(bool));
    s->max_val = max_val;
    return s;
}

int set_smallest_not_in(const struct set *s)
{
    for (size_t i = 0; i < s->max_val; i++)
    {
        if (!s->has_val[i])
        {
            return i;
        }
    }
    return -1;
}

// ============================================================================
// GRAPH UTILS
// ============================================================================

struct dict *degree_dict(const struct dict *qb_trm, size_t n_trm, size_t n_qb)
{
    struct dict *trm_deg = dict_new(n_trm, n_trm);
    for (size_t i = 0; i < n_trm; i++)
    {
        trm_deg->start[i] = i;
        trm_deg->length[i] = 1;
    }
    for (size_t qb = 0; qb < n_qb; qb++)
    {
        for (size_t i = 0; i < qb_trm->length[qb]; i++)
        {
            size_t ti = *dict_get(qb_trm, qb, i);
            int *p = (int *)dict_get(trm_deg, ti, 0);
            *p += qb_trm->length[qb] - 1;
        }
    }
    return trm_deg;
}

// ============================================================================
// COLORING FUNCTIONS
// ============================================================================

void degree_coloring(int *qb_idx,           // [in] flat array of qb indices on which terms act
                     size_t *trm_start_idx, // [in] start index of each term in qb_idx
                     size_t *n_qb_trm,      // [in] nb. of qubits on which each term acts
                     size_t n_trm,          // [in] nb. of terms
                     size_t n_qb,           // [in] nb. of qubits
                     int *trm_col           // [out] color of each term. array is already init.
)
{
    struct dict trm_qb = {.values = qb_idx, .start = trm_start_idx, .length = n_qb_trm, .n_keys = n_trm};
    struct dict *qb_trm = dict_invert(&trm_qb, n_qb);
    struct dict *trm_deg = degree_dict(&trm_qb, n_trm, n_qb);

    size_t max_deg = 0;
    for (size_t i = 0; i < n_trm; i++)
    {
        int d = *dict_get(trm_deg, i, 0);
        if (d > max_deg)
            max_deg = d;
    }
    struct dict *deg_trm = dict_invert(trm_deg, max_deg + 1);

    struct set *taken = set_new(max_deg + 1);
    for (size_t d = max_deg; d > 0; d--)
    {
        for (size_t i = 0; i < deg_trm->length[d]; i++)
        {
            size_t ti = *dict_get(deg_trm, d, i);
            memset(taken->has_val, 0, taken->max_val * sizeof(bool));
            for (size_t j = 0; j < trm_qb.length[ti]; j++)
            {
                int qb = *dict_get(&trm_qb, ti, j);
                for (size_t k = 0; k < qb_trm->length[qb]; k++)
                {
                    size_t other_trm_idx = *dict_get(qb_trm, qb, k);
                    if (other_trm_idx != ti)
                    {
                        set_add(taken, trm_col[other_trm_idx]);
                    }
                }
            }
            trm_col[ti] = set_smallest_not_in(taken);
        }
    }

    set_free(taken);
    dict_free(deg_trm);
    dict_free(trm_deg);
    dict_free(qb_trm);
}