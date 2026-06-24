#include <bits/stdc++.h>
#include "httplib.h"
#include "vector_db.h"
#include "document_db.h"

using namespace std;

// ════════════════════════════════════════════════════════
//  GLOBAL STATE
// ════════════════════════════════════════════════════════
struct Item {
    int id;
    string metadata;
    string category;
    vector<float> embedding;
};

vector<Item> itemStore;
int nextId = 1;
mutex storeMutex;

VectorDB db(16);
DocumentDB docdb;
OllamaClient ollama;

// ════════════════════════════════════════════════════════
//  HELPERS
// ════════════════════════════════════════════════════════
string read_file(const string& path) {
    ifstream file(path);
    if (!file.is_open()) return "";
    stringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

vector<float> random_vector(int dims) {
    random_device rd;
    mt19937 rng(rd());
    uniform_real_distribution<float> dist(-1.0, 1.0);
    vector<float> vec(dims);
    for (int i = 0; i < dims; i++) vec[i] = dist(rng);
    return vec;
}

// escape JSON strings
string json_escape(const string& s) {
    string out;
    for (char c : s) {
        if (c == '"')       out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else                out += c;
    }
    return out;
}

string vec_to_json(const vector<float>& v) {
    string s = "[";
    for (int i = 0; i < (int)v.size(); i++) {
        s += to_string(v[i]);
        if (i < (int)v.size() - 1) s += ",";
    }
    return s + "]";
}

string results_to_json(const vector<Result>& results, const string& category_hint = "") {
    string json = "[";
    for (int i = 0; i < (int)results.size(); i++) {
        // find item metadata
        string meta = results[i].id;
        string cat  = category_hint;
        lock_guard<mutex> lock(storeMutex);
        for (auto& item : itemStore) {
            if (to_string(item.id) == results[i].id) {
                meta = item.metadata;
                cat  = item.category;
                break;
            }
        }
        json += "{\"id\":" + results[i].id +
                ",\"metadata\":\"" + json_escape(meta) + "\"" +
                ",\"category\":\"" + cat + "\"" +
                ",\"distance\":" + to_string(results[i].distance) + "}";
        if (i < (int)results.size() - 1) json += ",";
    }
    return json + "]";
}

// ── JSON PARSERS ──
string parse_string(const string& body, const string& key) {
    string search = "\"" + key + "\":\"";
    size_t start = body.find(search);
    if (start == string::npos) return "";
    start += search.length();
    size_t end = start;
    while (end < body.size()) {
        if (body[end] == '\\') { end += 2; continue; }
        if (body[end] == '"') break;
        end++;
    }
    return body.substr(start, end - start);
}

int parse_int(const string& body, const string& key, int def = 5) {
    string search = "\"" + key + "\":";
    size_t start = body.find(search);
    if (start == string::npos) return def;
    start += search.length();
    size_t end = body.find_first_of(",}", start);
    if (end == string::npos) end = body.size();
    try { return stoi(body.substr(start, end - start)); }
    catch (...) { return def; }
}

vector<float> parse_vector_field(const string& body, const string& key) {
    string search = "\"" + key + "\":[";
    size_t start = body.find(search);
    vector<float> vec;
    if (start == string::npos) return vec;
    start += search.length();
    size_t end = body.find("]", start);
    if (end == string::npos) return vec;
    string nums = body.substr(start, end - start);
    stringstream ss(nums);
    string token;
    while (getline(ss, token, ',')) {
        try { vec.push_back(stof(token)); } catch (...) {}
    }
    return vec;
}

// parse comma-separated floats from query string
vector<float> parse_csv_floats(const string& s) {
    vector<float> vec;
    stringstream ss(s);
    string token;
    while (getline(ss, token, ',')) {
        try { vec.push_back(stof(token)); } catch (...) {}
    }
    return vec;
}

// ════════════════════════════════════════════════════════
//  SEED DEMO VECTORS (4 semantic clusters)
// ════════════════════════════════════════════════════════
void seed_demo_vectors() {
    struct Seed { string meta; string cat; vector<float> vec; };
    vector<Seed> seeds = {
        // CS — high dims 0-3
        {"binary_tree",    "cs",     {0.9f,0.8f,0.7f,0.6f, 0.1f,0.1f,0.1f,0.1f, 0.1f,0.1f,0.1f,0.1f, 0.1f,0.1f,0.1f,0.1f}},
        {"linked_list",    "cs",     {0.8f,0.9f,0.6f,0.7f, 0.1f,0.1f,0.1f,0.1f, 0.1f,0.1f,0.1f,0.1f, 0.1f,0.1f,0.1f,0.1f}},
        {"hash_table",     "cs",     {0.7f,0.6f,0.9f,0.8f, 0.1f,0.1f,0.1f,0.1f, 0.1f,0.1f,0.1f,0.1f, 0.1f,0.1f,0.1f,0.1f}},
        {"graph_bfs",      "cs",     {0.6f,0.7f,0.8f,0.9f, 0.1f,0.1f,0.1f,0.1f, 0.1f,0.1f,0.1f,0.1f, 0.1f,0.1f,0.1f,0.1f}},
        {"dynamic_prog",   "cs",     {0.8f,0.7f,0.9f,0.6f, 0.1f,0.1f,0.1f,0.1f, 0.1f,0.1f,0.1f,0.1f, 0.1f,0.1f,0.1f,0.1f}},
        // Math — high dims 4-7
        {"calculus",       "math",   {0.1f,0.1f,0.1f,0.1f, 0.9f,0.8f,0.7f,0.6f, 0.1f,0.1f,0.1f,0.1f, 0.1f,0.1f,0.1f,0.1f}},
        {"linear_algebra", "math",   {0.1f,0.1f,0.1f,0.1f, 0.8f,0.9f,0.6f,0.7f, 0.1f,0.1f,0.1f,0.1f, 0.1f,0.1f,0.1f,0.1f}},
        {"probability",    "math",   {0.1f,0.1f,0.1f,0.1f, 0.7f,0.6f,0.9f,0.8f, 0.1f,0.1f,0.1f,0.1f, 0.1f,0.1f,0.1f,0.1f}},
        {"statistics",     "math",   {0.1f,0.1f,0.1f,0.1f, 0.6f,0.7f,0.8f,0.9f, 0.1f,0.1f,0.1f,0.1f, 0.1f,0.1f,0.1f,0.1f}},
        {"number_theory",  "math",   {0.1f,0.1f,0.1f,0.1f, 0.9f,0.7f,0.8f,0.6f, 0.1f,0.1f,0.1f,0.1f, 0.1f,0.1f,0.1f,0.1f}},
        // Food — high dims 8-11
        {"sushi",          "food",   {0.1f,0.1f,0.1f,0.1f, 0.1f,0.1f,0.1f,0.1f, 0.9f,0.8f,0.7f,0.6f, 0.1f,0.1f,0.1f,0.1f}},
        {"pizza",          "food",   {0.1f,0.1f,0.1f,0.1f, 0.1f,0.1f,0.1f,0.1f, 0.8f,0.9f,0.6f,0.7f, 0.1f,0.1f,0.1f,0.1f}},
        {"ramen",          "food",   {0.1f,0.1f,0.1f,0.1f, 0.1f,0.1f,0.1f,0.1f, 0.7f,0.6f,0.9f,0.8f, 0.1f,0.1f,0.1f,0.1f}},
        {"biryani",        "food",   {0.1f,0.1f,0.1f,0.1f, 0.1f,0.1f,0.1f,0.1f, 0.6f,0.7f,0.8f,0.9f, 0.1f,0.1f,0.1f,0.1f}},
        {"pasta",          "food",   {0.1f,0.1f,0.1f,0.1f, 0.1f,0.1f,0.1f,0.1f, 0.8f,0.7f,0.9f,0.6f, 0.1f,0.1f,0.1f,0.1f}},
        // Sports — high dims 12-15
        {"basketball",     "sports", {0.1f,0.1f,0.1f,0.1f, 0.1f,0.1f,0.1f,0.1f, 0.1f,0.1f,0.1f,0.1f, 0.9f,0.8f,0.7f,0.6f}},
        {"cricket",        "sports", {0.1f,0.1f,0.1f,0.1f, 0.1f,0.1f,0.1f,0.1f, 0.1f,0.1f,0.1f,0.1f, 0.8f,0.9f,0.6f,0.7f}},
        {"football",       "sports", {0.1f,0.1f,0.1f,0.1f, 0.1f,0.1f,0.1f,0.1f, 0.1f,0.1f,0.1f,0.1f, 0.7f,0.6f,0.9f,0.8f}},
        {"tennis",         "sports", {0.1f,0.1f,0.1f,0.1f, 0.1f,0.1f,0.1f,0.1f, 0.1f,0.1f,0.1f,0.1f, 0.6f,0.7f,0.8f,0.9f}},
        {"badminton",      "sports", {0.1f,0.1f,0.1f,0.1f, 0.1f,0.1f,0.1f,0.1f, 0.1f,0.1f,0.1f,0.1f, 0.8f,0.7f,0.9f,0.6f}},
    };
    for (auto& s : seeds) {
        string id = to_string(nextId++);
        db.insert(id, s.vec);
        Item item;
        item.id = stoi(id);
        item.metadata = s.meta;
        item.category = s.cat;
        item.embedding = s.vec;
        itemStore.push_back(item);
    }
}

// ════════════════════════════════════════════════════════
//  MAIN
// ════════════════════════════════════════════════════════
int main() {
    // seed demo data
    seed_demo_vectors();

    httplib::Server svr;

    // ── CORS ──
    svr.set_post_routing_handler([](const auto& req, auto& res) {
        res.set_header("Access-Control-Allow-Origin",  "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
    });
    svr.Options(".*", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin",  "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
        res.status = 200;
    });

    // ════════════════════════════════════════════════════
    //  GET /  — serve index.html
    // ════════════════════════════════════════════════════
    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        string html = read_file("index.html");
        if (html.empty()) res.set_content("index.html not found", "text/plain");
        else res.set_content(html, "text/html");
    });

    // ════════════════════════════════════════════════════
    //  GET /items — return all stored items with embeddings
    // ════════════════════════════════════════════════════
    svr.Get("/items", [](const httplib::Request&, httplib::Response& res) {
        lock_guard<mutex> lock(storeMutex);
        string json = "[";
        for (int i = 0; i < (int)itemStore.size(); i++) {
            auto& item = itemStore[i];
            json += "{\"id\":" + to_string(item.id) +
                    ",\"metadata\":\"" + json_escape(item.metadata) + "\"" +
                    ",\"category\":\"" + item.category + "\"" +
                    ",\"embedding\":" + vec_to_json(item.embedding) + "}";
            if (i < (int)itemStore.size() - 1) json += ",";
        }
        json += "]";
        res.set_content(json, "application/json");
    });

    // ════════════════════════════════════════════════════
    //  GET /search?v=...&k=5&metric=cosine&algo=hnsw
    // ════════════════════════════════════════════════════
    svr.Get("/search", [](const httplib::Request& req, httplib::Response& res) {
        string v_str = req.has_param("v") ? req.get_param_value("v") : "";
        int k        = req.has_param("k") ? stoi(req.get_param_value("k")) : 5;
        string metric= req.has_param("metric") ? req.get_param_value("metric") : "cosine";
        string algo  = req.has_param("algo")   ? req.get_param_value("algo")   : "hnsw";

        auto query = parse_csv_floats(v_str);
        if (query.empty()) { res.set_content("{\"error\":\"invalid vector\"}", "application/json"); return; }

        auto t0 = chrono::high_resolution_clock::now();
        auto results = db.search(query, k, algo, metric);
        auto t1 = chrono::high_resolution_clock::now();
        long long us = chrono::duration_cast<chrono::microseconds>(t1-t0).count();

        string json = "{\"results\":" + results_to_json(results) + ",\"latencyUs\":" + to_string(us) + "}";
        res.set_content(json, "application/json");
    });

    // ════════════════════════════════════════════════════
    //  POST /insert — insert a new demo vector
    // ════════════════════════════════════════════════════
    svr.Post("/insert", [](const httplib::Request& req, httplib::Response& res) {
        string meta    = parse_string(req.body, "metadata");
        string cat     = parse_string(req.body, "category");
        auto embedding = parse_vector_field(req.body, "embedding");

        if (embedding.empty()) { res.set_content("{\"error\":\"invalid embedding\"}", "application/json"); return; }
        if (meta.empty()) meta = "unnamed";
        if (cat.empty())  cat  = "default";

        lock_guard<mutex> lock(storeMutex);
        string id = to_string(nextId);
        db.insert(id, embedding);

        Item item;
        item.id        = nextId++;
        item.metadata  = meta;
        item.category  = cat;
        item.embedding = embedding;
        itemStore.push_back(item);

        res.set_content("{\"id\":" + to_string(item.id) + ",\"status\":\"inserted\"}", "application/json");
    });

    // ════════════════════════════════════════════════════
    //  DELETE /delete/:id
    // ════════════════════════════════════════════════════
    svr.Delete(R"(/delete/(\d+))", [](const httplib::Request& req, httplib::Response& res) {
        int del_id = stoi(req.matches[1]);
        lock_guard<mutex> lock(storeMutex);
        auto it = find_if(itemStore.begin(), itemStore.end(), [&](const Item& i){ return i.id == del_id; });
        if (it == itemStore.end()) { res.set_content("{\"error\":\"not found\"}", "application/json"); return; }
        itemStore.erase(it);
        res.set_content("{\"status\":\"deleted\"}", "application/json");
    });

    // ════════════════════════════════════════════════════
    //  GET /benchmark?v=...&k=5&metric=cosine
    // ════════════════════════════════════════════════════
    svr.Get("/benchmark", [](const httplib::Request& req, httplib::Response& res) {
        string v_str = req.has_param("v") ? req.get_param_value("v") : "";
        int k        = req.has_param("k") ? stoi(req.get_param_value("k")) : 5;
        string metric= req.has_param("metric") ? req.get_param_value("metric") : "cosine";

        auto query = parse_csv_floats(v_str);
        if (query.empty()) { res.set_content("{\"error\":\"invalid vector\"}", "application/json"); return; }

        auto bench = [&](const string& algo) -> long long {
            auto t0 = chrono::high_resolution_clock::now();
            db.search(query, k, algo, metric);
            auto t1 = chrono::high_resolution_clock::now();
            return chrono::duration_cast<chrono::microseconds>(t1-t0).count();
        };

        long long bf = bench("bruteforce");
        long long kd = bench("kdtree");
        long long hn = bench("hnsw");

        string json = "{\"bruteforceUs\":" + to_string(bf) +
                      ",\"kdtreeUs\":"     + to_string(kd) +
                      ",\"hnswUs\":"       + to_string(hn) + "}";
        res.set_content(json, "application/json");
    });

    // ════════════════════════════════════════════════════
    //  GET /hnsw-info — layer stats
    // ════════════════════════════════════════════════════
    svr.Get("/hnsw-info", [](const httplib::Request&, httplib::Response& res) {
        int maxLayer    = db.get_hnsw_max_layer();
        int nodeCount   = db.get_node_count();

        // build nodes per layer and edges per layer
        string nodesArr = "[", edgesArr = "[";
        for (int l = maxLayer; l >= 0; l--) {
            int nodes = db.get_hnsw_nodes_at_layer(l);
            int edges = db.get_hnsw_edges_at_layer(l);
            nodesArr += to_string(nodes);
            edgesArr += to_string(edges);
            if (l > 0) { nodesArr += ","; edgesArr += ","; }
        }
        nodesArr += "]"; edgesArr += "]";

        string json = "{\"maxLayer\":" + to_string(maxLayer) +
                      ",\"nodeCount\":" + to_string(nodeCount) +
                      ",\"nodesPerLayer\":" + nodesArr +
                      ",\"edgesPerLayer\":" + edgesArr + "}";
        res.set_content(json, "application/json");
    });

    // ════════════════════════════════════════════════════
    //  GET /status — Ollama status
    // ════════════════════════════════════════════════════
    svr.Get("/status", [](const httplib::Request&, httplib::Response& res) {
        bool online = ollama.is_online();
        int docCount = docdb.size();
        string json = "{\"ollamaAvailable\":" + string(online ? "true" : "false") +
                      ",\"embedModel\":\"nomic-embed-text\"" +
                      ",\"genModel\":\"llama3.2\"" +
                      ",\"docCount\":" + to_string(docCount) +
                      ",\"docDims\":768}";
        res.set_content(json, "application/json");
    });

    // ════════════════════════════════════════════════════
    //  POST /doc/insert — embed and store document
    // ════════════════════════════════════════════════════
    svr.Post("/doc/insert", [](const httplib::Request& req, httplib::Response& res) {
        string title = parse_string(req.body, "title");
        string text  = parse_string(req.body, "text");

        if (title.empty() || text.empty()) {
            res.set_content("{\"error\":\"need title and text\"}", "application/json"); return;
        }

        int before = docdb.size();
        docdb.add_document(title, text);
        int after = docdb.size();
        int chunks = after - before;

        string json = "{\"status\":\"ok\",\"chunks\":" + to_string(chunks) + ",\"dims\":768}";
        res.set_content(json, "application/json");
    });

    // ════════════════════════════════════════════════════
    //  GET /doc/list — list all document chunks
    // ════════════════════════════════════════════════════
    svr.Get("/doc/list", [](const httplib::Request&, httplib::Response& res) {
        auto docs = docdb.get_all();
        string json = "[";
        for (int i = 0; i < (int)docs.size(); i++) {
            string preview = docs[i].text.substr(0, 120);
            // count words
            int words = 0;
            bool inWord = false;
            for (char c : docs[i].text) {
                if (isspace(c)) inWord = false;
                else if (!inWord) { inWord = true; words++; }
            }
            json += "{\"id\":\"" + docs[i].id + "\"" +
                    ",\"title\":\"" + json_escape(docs[i].title) + "\"" +
                    ",\"preview\":\"" + json_escape(preview) + "\"" +
                    ",\"words\":" + to_string(words) + "}";
            if (i < (int)docs.size() - 1) json += ",";
        }
        json += "]";
        res.set_content(json, "application/json");
    });

    // ════════════════════════════════════════════════════
    //  DELETE /doc/delete/:id
    // ════════════════════════════════════════════════════
    svr.Delete(R"(/doc/delete/(.+))", [](const httplib::Request& req, httplib::Response& res) {
        string del_id = req.matches[1];
        docdb.delete_document(del_id);
        res.set_content("{\"status\":\"deleted\"}", "application/json");
    });

    // ════════════════════════════════════════════════════
    //  POST /doc/search — semantic search over documents
    // ════════════════════════════════════════════════════
    svr.Post("/doc/search", [](const httplib::Request& req, httplib::Response& res) {
        string question = parse_string(req.body, "question");
        int k = parse_int(req.body, "k", 3);
        if (question.empty()) { res.set_content("{\"error\":\"empty question\"}", "application/json"); return; }

        auto docs = docdb.search(question, k);
        string json = "{\"contexts\":[";
        for (int i = 0; i < (int)docs.size(); i++) {
            json += "{\"title\":\"" + json_escape(docs[i].title) + "\"" +
                    ",\"text\":\"" + json_escape(docs[i].text.substr(0, 300)) + "\"" +
                    ",\"id\":\"" + docs[i].id + "\"" +
                    ",\"distance\":0.0}";
            if (i < (int)docs.size() - 1) json += ",";
        }
        json += "]}";
        res.set_content(json, "application/json");
    });

    // ════════════════════════════════════════════════════════
    //  POST /doc/ask — RAG pipeline (falls back to LLM if no docs)
    // ════════════════════════════════════════════════════════
    svr.Post("/doc/ask", [](const httplib::Request& req, httplib::Response& res) {
        string question = parse_string(req.body, "question");
        int k = parse_int(req.body, "k", 3);
        if (question.empty()) { res.set_content("{\"error\":\"empty question\"}", "application/json"); return; }

        string prompt;
        string json;

        if (docdb.size() == 0) {
            // No documents — answer purely from LLM knowledge
            prompt = "You are a helpful assistant. Answer the following question:\n\nQuestion: " + question + "\n\nAnswer:";
            string answer = ollama.generate(prompt);
            json = "{\"answer\":\"" + json_escape(answer) + "\"" +
                ",\"model\":\"llama3.2\"" +
                ",\"contexts\":[]}";
        } else {
            // RAG path — retrieve top-k chunks and build context
            auto docs = docdb.search(question, k);

            string context = "";
            for (auto& d : docs) context += d.text + "\n\n";

            prompt = "You are a helpful assistant. Use the context below if relevant. "
                    "If the context does not contain the answer, use your own knowledge.\n\n"
                    "Context:\n" + context +
                    "\nQuestion: " + question + "\n\nAnswer:";

            string answer = ollama.generate(prompt);

            json = "{\"answer\":\"" + json_escape(answer) + "\"" +
                ",\"model\":\"llama3.2\"" +
                ",\"contexts\":[";
            for (int i = 0; i < (int)docs.size(); i++) {
                string preview = docs[i].text.substr(0, 300);
                json += "{\"title\":\"" + json_escape(docs[i].title) + "\"" +
                        ",\"text\":\"" + json_escape(preview) + "\"" +
                        ",\"distance\":0." + to_string(i+1) + "}";
                if (i < (int)docs.size() - 1) json += ",";
            }
            json += "]}";
        }

        res.set_content(json, "application/json");
    });

    // ════════════════════════════════════════════════════
    //  BOOT
    // ════════════════════════════════════════════════════
    cout << "  http://localhost:5000\n";
    cout << "  " << itemStore.size() << " demo vectors | 16 dims | HNSW + KD-Tree + BruteForce\n";
    cout << "  Ollama: " << (ollama.is_online() ? "ONLINE" : "OFFLINE") << "\n";
    cout << "\n";

    svr.listen("0.0.0.0", 5000);
    return 0;
}