#pragma once
#include "httplib.h"
#include <bits/stdc++.h>

using namespace std;

class OllamaClient {
    string embed_model;
    string gen_model;
    string host;
    int port;

public:
    OllamaClient() {
        this -> embed_model = "nomic-embed-text";
        this -> gen_model = "llama3.2";
        this -> host = "localhost";
        this -> port = 3000;
    }

    vector<float> embed(const string& text) {
        httplib::Client client(host, port);

        string body = "{\"model\":\"" + embed_model + "\",\"prompt\":\"" + text + "\"}";
        
        auto res = client.Post("/api/embeddings", body, "application/json");

        vector<float> embedding;

        if(!res || res -> status != 200) return embedding;

        string resp = res -> body;

        int start = -1;
        int end = -1;

        for(int i = 0; i < resp.size(); i++) {
            if(resp[i] == '[') start = i;
            else if(resp[i] == ']') end = i;
        }

        if(start == -1 || end == -1) return embedding;

        string nums = resp.substr(start + 1, end - start - 1);

        stringstream ss(nums);
        string token;

        while(getline(ss, token, ',')) {
            embedding.push_back(stof(token));
        }

        return embedding;
    }

    string generate(const string& prompt) {
        httplib::Client client(host, port);

        string body = "{\"model\":\"" + gen_model + "\",\"prompt\":\"" + prompt + "\",\"stream\":false}";

        auto res = client.Post("/api/generate", body, "application/json");

        if(!res || res->status != 200) return "Ollama error";

        string resp = res->body;

        string key = "\"response\":\"";
        
        int start = resp.find(key);

        if(start == string::npos) return "parse error";

        start += key.length();
        int end = resp.find("\"", start);

        return resp.substr(start, end - start);
    }

    bool is_online() {
        httplib::Client client(host, port);
        auto res = client.Get("/");
        return res && res->status == 200;
    }
};