#pragma once
#include "brute_force.h"
#include "kdtree.h"
#include "hnsw.h"

class VectorDB {
    BruteForce bf;
    KDTree kdtree;
    HNSW hnsw;
    int dims;

public:
    VectorDB(int dims) : dims(dims), kdtree(dims) {}

    void insert(const string& id, const vector<float>& vec) {
        bf.insert(id, vec);
        kdtree.insert(id, vec);
        hnsw.insert(id, vec);
    }

    vector<Result> search(const vector<float>& query, int k, const string& algo, const string& metric) {
        if(algo == "hnsw") return hnsw.search(query, k);
        else if(algo == "kdtree") return kdtree.search(query, k, metric);
        else return bf.search(query, k, metric);
    }

    int get_node_count() {
        return hnsw.get_node_count();
    }
};