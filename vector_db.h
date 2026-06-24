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
        if (algo == "hnsw")       return hnsw.search(query, k);
        else if (algo == "kdtree") return kdtree.search(query, k, metric);
        else                       return bf.search(query, k, metric);
    }

    int get_node_count()            { return hnsw.get_node_count(); }
    int get_hnsw_max_layer()        { return hnsw.get_max_layer(); }

    // nodes at a given layer
    int get_hnsw_nodes_at_layer(int layer) {
        return hnsw.get_nodes_at_layer(layer);
    }

    // edges at a given layer
    int get_hnsw_edges_at_layer(int layer) {
        return hnsw.get_edges_at_layer(layer);
    }
};