#pragma once
#include "structs.h"
#include "math_utils.h"
#include <bits/stdc++.h>

class HNSW {
    unordered_map<string, HNSWNode> nodes;
    string entry_point;
    int max_layer;
    int M;
    int ef_construction;
    mt19937 rng;

    int get_random_layer() {
        uniform_real_distribution<float> dist(0.0, 1.0);
        float r = dist(rng);
        if(r <= 0.0f) r = 1e-10f;

        // Exponential Distribution So that more layers at 0th level and Lesser at Above Layers
        int layer = (int)(-log(r) * (1.0 / log(M)));
        return min(layer, 6);
    }

    vector<pair<float,string>> search_layer(const vector<float>& query, const string& ep, int ef, int layer) {
        unordered_set<string> visited;

        priority_queue<pair<float,string>, vector<pair<float,string>>, greater<pair<float,string>>> candidates;

        priority_queue<pair<float,string>> results;

        float d = distance(query, nodes[ep].vec, "cosine");
        candidates.push({d, ep});
        results.push({d, ep});
        visited.insert(ep);

        while(!candidates.empty()) {
            auto top = candidates.top();
            candidates.pop();

            float cd = top.first;
            string cid = top.second;
            
            if(cd > results.top().first) break;

            HNSWNode curNode = nodes[cid];
            
            for(auto &nbr : curNode.neighbors[layer]) {
                if(visited.count(nbr) == 1) continue;

                visited.insert(nbr);

                float dist = distance(query, nodes[nbr].vec, "cosine");
                
                if(dist < results.top().first || (int)results.size() < ef) {
                    candidates.push({dist, nbr});
                    results.push({dist, nbr});

                    if((int)results.size() > ef) results.pop();
                }
            }
        }

        vector<pair<float, string>> answer;
        
        while(!results.empty()) {
            answer.push_back(results.top());
            results.pop();
        }

        return answer;    
    }

    vector<string> select_neighbors(vector<pair<float,string>>& candidates, int M) {
        int size = candidates.size();

        sort(candidates.begin(), candidates.end());

        vector<string> topMIds;
        for(int i = 0;i < min(size, M); i++) {
            topMIds.push_back(candidates[i].second);
        }

        return topMIds;
    }

public:
    HNSW(int M = 16, int ef_construction = 200)
        : M(M), ef_construction(ef_construction), max_layer(0), rng(random_device{}()) {}

    void insert(const string& id, const vector<float>& vec) {
        int new_layer = get_random_layer();

        HNSWNode node(vec, id, new_layer);

        if(nodes.empty()) {
            entry_point = id;
            max_layer = new_layer;
            nodes[id] = node;
            return;
        }

        nodes[id] = node;
        string ep = entry_point;

        for(int layer = max_layer; layer > new_layer; layer--) {
            auto results = search_layer(vec, ep, 1, layer);
            ep = results[0].second;
        }

        for(int layer = min(new_layer, max_layer); layer >= 0; layer--) {
            auto results = search_layer(vec, ep, ef_construction, layer);
            
            auto neighbors = select_neighbors(results, M);
            
            nodes[id].neighbors[layer] = neighbors;
            
            for(auto& nb_id : neighbors) {
                nodes[nb_id].neighbors[layer].push_back(id);
                
                // triming if exceeded M
                if((int)nodes[nb_id].neighbors[layer].size() > M) {
                    vector<pair<float,string>> nb_candidates;
                    
                    for(auto& nb_nb : nodes[nb_id].neighbors[layer]) {
                        float d = distance(nodes[nb_id].vec, nodes[nb_nb].vec, "cosine");
                        nb_candidates.push_back({d, nb_nb});
                    }
                    
                    nodes[nb_id].neighbors[layer] = select_neighbors(nb_candidates, M);
                }
            }
            
            if(!results.empty()) ep = results[0].second;
        }

        if(new_layer > max_layer) {
            max_layer = new_layer;
            entry_point = id;
        }
    }

    vector<Result> search(const vector<float>& query, int k, int ef = 50) {
        if(nodes.empty()) return {};

        string ep = entry_point;

        // layers above 0 → greedy zoom, ef=1
        for(int layer = max_layer; layer > 0; layer--) {
            auto results = search_layer(query, ep, 1, layer);
            ep = results[0].second;
        }

        // layer 0 → full beam search
        auto results = search_layer(query, ep, max(ef, k), 0);

        sort(results.begin(), results.end());

        vector<Result> out;
        for(int i = 0; i < min(k, (int)results.size()); i++) {
            out.push_back({results[i].first, results[i].second});
        }

        return out;
    }

    int get_max_layer() { return max_layer; }
    int get_node_count() { return nodes.size(); }
    string get_entry_point() { return entry_point; }
};