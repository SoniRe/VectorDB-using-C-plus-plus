#pragma once
#include "structs.h"
#include "math_utils.h"

struct KDNode {
    vector<float> vec;
    string id;
    KDNode* left;
    KDNode* right;

    KDNode(vector<float> vec, string id) {
        this -> vec = vec;
        this -> id = id;
        this -> left = NULL;
        this -> right = NULL;
    }
};

class KDTree {
    KDNode* root;
    int dims; // Number of dimensions

    KDNode* insert_recursive(KDNode* node, const vector<float>& vec, const string& id, int depth) {
        if(node == NULL) {
            KDNode *newNode = new KDNode(vec, id);
            return newNode;
        }

        int curDim = depth%dims;

        if(vec[curDim] < node -> vec[curDim]) {
            node -> left = insert_recursive(node -> left, vec, id, depth + 1);
        }
        else {
            node -> right = insert_recursive(node -> right, vec, id, depth + 1);
        }

        return node;
    }

    void search_recursive(KDNode* node, const vector<float>& query, int k, int depth, priority_queue<pair<float,string>>& pq) {
        if(node == NULL) return;

        float dist = distance(query, node -> vec, "euclidean");
        
        pq.push({dist, node -> id});
        
        if(pq.size() > k) pq.pop();
        
        int curDim = depth%dims;

        bool checkedLeft = true;

        if(query[curDim] < node -> vec[curDim]) {
            search_recursive(node -> left, query, k, depth + 1, pq);
        }
        else {
            search_recursive(node -> right, query, k, depth + 1, pq);
            checkedLeft = false;
        }

        float distToLine = abs(query[curDim] - node -> vec[curDim]);

        if(distToLine < pq.top().first) {
            if(checkedLeft) search_recursive(node -> right, query, k, depth + 1, pq);
            else search_recursive(node -> left, query, k, depth + 1, pq);
        }
    }

public:
    KDTree(int dims) : root(nullptr), dims(dims) {}

    void insert(const string& id, const vector<float>& vec) {
        root = insert_recursive(root, vec, id, 0);
    }

    vector<Result> search(const vector<float>& query, int k, const string& metric) {
        priority_queue<pair<float,string>> pq;

        search_recursive(root, query, k, 0, pq);

        vector<Result> ans;
        while(!pq.empty()) {
            auto front = pq.top();
            pq.pop();

            Result res;
            res.distance = front.first;
            res.id = front.second;

            ans.push_back(res);
        }

        sort(ans.begin(), ans.end());

        return ans;
    }
};