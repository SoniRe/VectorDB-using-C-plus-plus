#pragma once
#include <bits/stdc++.h>

using namespace std;

struct Result {
    float distance;
    string id;

    // so we can sort by distance
    bool operator<(const Result& other) const {
        return distance < other.distance;
    }
};

struct Document {
    string id;
    string title;
    string text;
    vector<float> embedding;

    Document() {}
    Document(string id, string title, string text, vector<float> embedding) {
        this -> id = id;
        this -> title = title;
        this -> text = text;
        this -> embedding = embedding;
    }
};

struct HNSWNode {
    vector<float> vec;
    string id;
    vector<vector<string>> neighbors;
    int max_layer;

    HNSWNode() {}
    HNSWNode(vector<float> vec, string id, int max_layer) {
        this -> vec = vec;
        this -> id = id;
        this -> max_layer = max_layer;
        this -> neighbors.resize(max_layer + 1);
    }
};