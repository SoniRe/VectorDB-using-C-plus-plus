#pragma once
#include "httplib.h"
#include <bits/stdc++.h>

using namespace std;

class OllamaClient {
    string embed_model;
    string gen_model;
    string host;
    int port;

    // Escapes a plain string into a safe JSON string value (no surrounding quotes)
    string json_escape(const string& s) {
        string out;
        for (unsigned char c : s) {
            switch (c) {
                case '"':  out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\n': out += "\\n";  break;
                case '\r': out += "\\r";  break;
                case '\t': out += "\\t";  break;
                default:
                    if (c < 0x20) {
                        // control characters → \uXXXX
                        char buf[8];
                        snprintf(buf, sizeof(buf), "\\u%04x", c);
                        out += buf;
                    } else {
                        out += c;
                    }
            }
        }
        return out;
    }

    // Extracts the value of a JSON string field, correctly handling escaped quotes inside the value
    string parse_json_string(const string& resp, const string& key) {
        string search = "\"" + key + "\":\"";
        size_t start = resp.find(search);
        if (start == string::npos) return "";
        start += search.length();

        string result;
        size_t i = start;
        while (i < resp.size()) {
            if (resp[i] == '\\' && i + 1 < resp.size()) {
                // handle escape sequences
                char next = resp[i + 1];
                switch (next) {
                    case '"':  result += '"';  break;
                    case '\\': result += '\\'; break;
                    case 'n':  result += '\n'; break;
                    case 'r':  result += '\r'; break;
                    case 't':  result += '\t'; break;
                    default:   result += next; break;
                }
                i += 2;
            } else if (resp[i] == '"') {
                // unescaped quote = end of string value
                break;
            } else {
                result += resp[i];
                i++;
            }
        }
        return result;
    }

public:
    OllamaClient() {
        this -> embed_model = "nomic-embed-text";
        this -> gen_model = "llama3.2";
        this -> host = "localhost";
        this -> port = 11434;
    }

    vector<float> embed(const string& text) {
        
        httplib::Client client(host, port);
        client.set_connection_timeout(10, 0);
        client.set_read_timeout(30, 0);

        // FIX: escape the text so special characters don't break the JSON body
        string body = "{\"model\":\"" + embed_model + "\",\"prompt\":\"" + json_escape(text) + "\"}";
        
        auto res = client.Post("/api/embeddings", body, "application/json");

        vector<float> embedding;

        if(!res || res -> status != 200) return embedding;

        string resp = res -> body;

        int start = -1;
        int end = -1;

        for(int i = 0; i < (int)resp.size(); i++) {
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
        client.set_connection_timeout(10, 0);
        client.set_read_timeout(120, 0);
        
        // FIX: escape the prompt so quotes/newlines/backslashes don't corrupt the JSON body
        string body = "{\"model\":\"" + gen_model + "\",\"prompt\":\"" + json_escape(prompt) + "\",\"stream\":false}";

        auto res = client.Post("/api/generate", body, "application/json");

        if(!res || res->status != 200) return "Ollama error";

        // FIX: use escape-aware parser so answers containing \" don't get truncated
        return parse_json_string(res->body, "response");
    }

    bool is_online() {
        httplib::Client client(host, port);
        client.set_connection_timeout(3, 0);
        auto res = client.Get("/api/tags");
        return res && res->status == 200;
    }
};