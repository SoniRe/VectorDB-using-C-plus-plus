#pragma once
#include "structs.h"
#include "math_utils.h"

struct KDNode {
    vector<float> vec;
    string id;
    KDNode* left;
    KDNode* right;
};

class KDTree {
    KDNode* root;
    int dims; // Number of dimensions

    KDNode* insert_recursive(KDNode* node, const vector<float>& vec, const string& id, int depth) {
        // 1. if node is null → create new KDNode and return it
        if(node == NULL) return node;
        // 2. find current dimension → depth % dims
        int curDim = depth%dims;

        // 3. if vec[dim] < node->vec[dim] → go left
        // 4. else → go right
        // 5. return node
    }

    void search_recursive(KDNode* node, const vector<float>& query, int k, int depth, priority_queue<pair<float,string>>& pq) {
        // 1. if node is null → return
        if(node == NULL) return;
        // 2. calculate distance(query, node->vec)
        // 3. push into pq
        // 4. if pq size > k → pop largest
        // 5. find current dimension → depth % dims
        // 6. decide which side to search first
        // 7. recursively search that side
        // 8. check if other side could have closer points
        // 9. if yes → search other side too
    }

public:
    KDTree(int dims) : root(nullptr), dims(dims) {}

    void insert(const string& id, const vector<float>& vec) {
        // call insert_recursive
        insert_recursive(root, vec, id, 0);
    }

    vector<Result> search(const vector<float>& query, int k, const string& metric) {
        // 1. create priority_queue
        // 2. call search_recursive
        // 3. convert pq to vector<Result>
        // 4. sort and return
    }
};