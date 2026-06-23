#include <bits/stdc++.h>
#include "httplib.h"
#include "vector_db.h"
#include "document_db.h"

using namespace std;

string read_file(const string& path) {
    ifstream file(path);
    if(!file.is_open()) return "";
    stringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

vector<float> random_vector(int dims) {
    random_device rd;
    mt19937 rng(rd());
    uniform_real_distribution<float> dist(-1.0, 1.0);
    vector<float> vec(dims);
    for(int i = 0; i < dims; i++) vec[i] = dist(rng);
    return vec;
}

string results_to_json(const vector<Result>& results) {
    string json = "[";
    for(int i = 0; i < results.size(); i++) {
        json += "{\"id\":\"" + results[i].id + "\",\"distance\":" + to_string(results[i].distance) + "}";
        if(i < results.size() - 1) json += ",";
    }
    json += "]";
    return json;
}

string parse_string(const string& body, const string& key) {
    string search = "\"" + key + "\":\"";
    int start = body.find(search);
    if(start == string::npos) return "";
    start += search.length();
    int end = body.find("\"", start);
    return body.substr(start, end - start);
}

int parse_int(const string& body, const string& key) {
    string search = "\"" + key + "\":";
    int start = body.find(search);
    if(start == string::npos) return 5;
    start += search.length();
    int end = body.find_first_of(",}", start);
    return stoi(body.substr(start, end - start));
}

vector<float> parse_vector(const string& body, const string& key) {
    string search = "\"" + key + "\":[";
    int start = body.find(search);
    vector<float> vec;
    if(start == string::npos) return vec;
    start += search.length();
    int end = body.find("]", start);
    string nums = body.substr(start, end - start);
    stringstream ss(nums);
    string token;
    while(getline(ss, token, ',')) vec.push_back(stof(token));
    return vec;
}

int main() {
    VectorDB db(16);
    DocumentDB docdb;

    // seed 20 random vectors
    for(int i = 0; i < 20; i++) {
        string id = "v" + to_string(i + 1);
        db.insert(id, random_vector(16));
    }

    httplib::Server svr;

    // CORS
    svr.set_post_routing_handler([](const auto& req, auto& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
    });

    svr.Options(".*", [](const httplib::Request&, httplib::Response& res) {
        res.status = 200;
    });

    // serve index.html
    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        string html = read_file("index.html");
        if(html.empty()) res.set_content("index.html not found", "text/plain");
        else res.set_content(html, "text/html");
    });

    // POST /search
    svr.Post("/search", [&](const httplib::Request& req, httplib::Response& res) {
        auto query = parse_vector(req.body, "query");
        int k = parse_int(req.body, "k");
        string algo = parse_string(req.body, "algo");
        string metric = parse_string(req.body, "metric");

        if(query.empty()) {
            res.set_content("{\"error\":\"invalid query\"}", "application/json");
            return;
        }

        auto results = db.search(query, k, algo, metric);
        string json = "{\"results\":" + results_to_json(results) + "}";
        res.set_content(json, "application/json");
    });

    // POST /insert
    svr.Post("/insert", [&](const httplib::Request& req, httplib::Response& res) {
        string id = parse_string(req.body, "id");
        auto vec = parse_vector(req.body, "vector");

        if(id.empty() || vec.empty()) {
            res.set_content("{\"error\":\"invalid input\"}", "application/json");
            return;
        }

        db.insert(id, vec);
        res.set_content("{\"status\":\"inserted\"}", "application/json");
    });

    // GET /vectors
    svr.Get("/vectors", [&](const httplib::Request&, httplib::Response& res) {
        string json = "{\"count\":" + to_string(db.get_node_count()) + "}";
        res.set_content(json, "application/json");
    });

    // GET /status
    svr.Get("/status", [&](const httplib::Request&, httplib::Response& res) {
        OllamaClient ollama;
        bool online = ollama.is_online();
        string json = "{\"ollama\":" + string(online ? "true" : "false") + "}";
        res.set_content(json, "application/json");
    });

    // POST /documents
    svr.Post("/documents", [&](const httplib::Request& req, httplib::Response& res) {
        string title = parse_string(req.body, "title");
        string text = parse_string(req.body, "text");

        if(title.empty() || text.empty()) {
            res.set_content("{\"error\":\"invalid input\"}", "application/json");
            return;
        }

        docdb.add_document(title, text);
        res.set_content("{\"status\":\"added\",\"chunks\":" + to_string(docdb.size()) + "}", "application/json");
    });

    // GET /documents
    svr.Get("/documents", [&](const httplib::Request&, httplib::Response& res) {
        auto docs = docdb.get_all();
        string json = "[";
        for(int i = 0; i < docs.size(); i++) {
            json += "{\"id\":\"" + docs[i].id + "\",\"title\":\"" + docs[i].title + "\",\"text\":\"" + docs[i].text.substr(0, 100) + "\"}";
            if(i < docs.size() - 1) json += ",";
        }
        json += "]";
        res.set_content(json, "application/json");
    });

    // POST /ask
    svr.Post("/ask", [&](const httplib::Request& req, httplib::Response& res) {
        string question = parse_string(req.body, "question");

        if(question.empty()) {
            res.set_content("{\"error\":\"invalid question\"}", "application/json");
            return;
        }

        string answer = docdb.ask(question);
        res.set_content("{\"answer\":\"" + answer + "\"}", "application/json");
    });

    cout << "VecDB running on http://localhost:8080" << endl;
    svr.listen("0.0.0.0", 8080);

    return 0;
}