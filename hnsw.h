#pragma once
#include "structs.h"
#include "math_utils.h"

struct HNSWNode {
    vector<float> vec;
    string id;
    vector<vector<string>> neighbors; // neighbors[layer] = list of ids
    int max_layer;
};

class HNSW {
    unordered_map<string, HNSWNode> nodes;
    string entry_point; // id of entry node
    int max_layer;      // current max layer in graph
    int M;              // max neighbors per layer
    int ef_construction;// beam width during insert
    mt19937 rng;        // random number generator

    int get_random_layer() {
        // randomly assign a layer using exponential distribution
        // hint: use uniform_real_distribution and log
    }

    vector<pair<float,string>> search_layer(const vector<float>& query, const string& ep, int ef, int layer) {
        // beam search on a single layer
        // returns ef closest candidates
    }

    vector<string> select_neighbors(const vector<float>& query, vector<pair<float,string>>& candidates, int M) {
        // pick best M neighbors from candidates
        // sort by distance, return top M ids
    }

public:
    HNSW(int M = 16, int ef_construction = 200)
        : M(M), ef_construction(ef_construction), max_layer(0), rng(random_device{}()) {}

    void insert(const string& id, const vector<float>& vec) {
        // 1. get random layer for new node
        // 2. create HNSWNode
        // 3. if graph empty → set as entry point
        // 4. find closest entry points layer by layer
        // 5. connect new node to neighbors at each layer
    }

    vector<Result> search(const vector<float>& query, int k, int ef = 50) {
        // 1. start from entry point
        // 2. search from top layer down to layer 0
        // 3. at layer 0 use ef as beam width
        // 4. return top k results
    }
};