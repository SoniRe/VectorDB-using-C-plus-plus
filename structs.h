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