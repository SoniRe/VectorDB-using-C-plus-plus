#pragma once
#include "structs.h"
#include "math_utils.h"
#include "ollama.h"

class DocumentDB {
    vector<Document> docs;
    OllamaClient ollama;
    int chunk_size;
    int doc_counter;

    vector<string> chunk_text(const string& text) {
        vector<string> chunks;
        stringstream ss(text);
        string word;
        vector<string> words;

        while(ss >> word) {
            words.push_back(word);
        }

        string chunk = "";
        int count = 0;

        for(int i = 0; i < words.size(); i++) {
            chunk += words[i] + " ";
            count++;

            if(count == chunk_size) {
                chunks.push_back(chunk);
                chunk = "";
                count = 0;
            }
        }

        if(chunk != "") chunks.push_back(chunk);

        return chunks;
    }

public:
    DocumentDB() {
        this -> chunk_size = 200;
        this -> doc_counter = 0;
    }

    void add_document(const string& title, const string& text) {
        // 1. chunk the text
        vector<string> chunks = chunk_text(text);
        // 2. for each chunk:
        for(auto &chunk : chunks) {
            string id = "doc_" + to_string(doc_counter++);
            vector<float> embedding = ollama.embed(chunk);

            Document doc(id, title, chunk, embedding);
            docs.push_back(doc);
        }
    }

    vector<Document> search(const string& query, int k) {
        vector<float> query_embedding = ollama.embed(query);
      
        priority_queue<pair<float, int>> pq;

        for(int i = 0;i < docs.size(); i++) {
            float similarity = cosine_similarity(query_embedding, docs[i].embedding);
            pq.push({similarity, i});
        }

        vector<Document> topKDocs;

        while(k > 0 && !pq.empty()) {
            topKDocs.push_back(docs[pq.top().second]);
            pq.pop();
            k--;
        }

        return topKDocs;
    }

    string ask(const string& question, int k = 3) {
        vector<Document> topKChunks = search(question, k);
        
        string context = "";
        
        for(auto& doc : topKChunks) {
            context += doc.text + " ";
        }

        string prompt = "Context: " + context + "\n\nQuestion: " + question + "\n\nAnswer based only on the context above:";

        string answer = ollama.generate(prompt);

        return answer;
    }

    vector<Document> get_all() {
        return docs;
    }

    int size() {
        return docs.size();
    }
};