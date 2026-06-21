#include <bits/stdc++.h>
using namespace std;

float dot(const vector<float>& a, const vector<float>& b) {
    int size = a.size();
    float ans = 0;

    for(int i = 0;i < size; i++) 
    ans += (a[i]*b[i]);

    return ans;
}

float norm(const vector<float>& a) {
    double ans = 0;

    for(auto el : a) ans += (el * el);

    return sqrt(ans);
}

float cosine_similarity(const vector<float>& a, const vector<float>& b) {
    float dot_a_b = dot(a, b);
    float norm_a = norm(a);
    float norm_b = norm(b);

    return dot_a_b / (norm_a * norm_b);
}

float euclidean(const vector<float>& a, const vector<float>& b) {
    float ans = 0;
    int size = a.size();

    for(int i = 0;i < size; i++) {
        float diff_sqrd = (a[i] - b[i]) * (a[i] - b[i]);

        ans += diff_sqrd;
    }   

    return sqrt(ans);
}


float manhattan(const vector<float>& a, const vector<float>& b) {
    float ans = 0;
    int size = a.size();

    for(int i = 0;i < size; i++) {
        ans += abs(a[i] - b[i]);
    }

    return ans;
}


float distance(const vector<float>& a, const vector<float>& b, const string& metric) {
    float ans = 0;

    if(metric == "cosine") {
        ans = 1 - cosine_similarity(a, b);
    }
    else if(metric == "euclidean") {
        ans = euclidean(a, b);
    }
    else {
        // Default Choice
        ans = manhattan(a, b);
    }

    return ans;
}