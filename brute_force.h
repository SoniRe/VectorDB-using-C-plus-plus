#pragma once
#include "structs.h"
#include "math_utils.h"

class BruteForce {
    unordered_map<string, vector<float>> store;

public:
    void insert(const string& id, const vector<float>& vec) {
        store[id] = vec;
    }
    
    vector<Result> search(const vector<float>& query, int k, const string& metric) {
        vector<Result> results;

        for(auto &entry : store) {
            float dist = distance(query, entry.second, metric);

            Result res;
            res.distance = dist;
            res.id = entry.first;
            
            results.push_back(res);
        }

        sort(results.begin(), results.end());

        // Return Top - k
        int size = results.size();
        results.resize(min(k, size));
        return results;
    }
};