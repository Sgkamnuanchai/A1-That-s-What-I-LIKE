from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
import json
from Model.skipgramNec_model import SkipgramNegSampling
app = Flask(__name__)

# Load vocabulary
with open("./Model/json/word2index.json", "r") as f:
    word2index = json.load(f)

with open("./Model/json/vocabs.json", "r") as f:
    vocabs = json.load(f)


vocab_size = len(word2index)
embedding_size = 2
model = SkipgramNegSampling(vocab_size, embedding_size)
# Load the entire model
model_path = "./Model/model_skipgram_neg.pth"
model.load_state_dict(torch.load(model_path))
# Set the model to evaluation mode
model.eval()

def get_embed(model, word):
    try:
        index = word2index[word]
    except:
        index = word2index['<UNK>']
    
    word = torch.LongTensor([word2index[word]])
    
    embed_c = model.embedding_v(word)
    embed_o = model.embedding_u(word)
    embed   = (embed_c + embed_o) / 2
    
    return embed[0][0].item(), embed[0][1].item()

def cosine_similarity(A, B):
    dot_product = np.dot(A.flatten(), B.flatten())
    norm_a = np.linalg.norm(A)
    norm_b = np.linalg.norm(B)
    similarity = dot_product / (norm_a * norm_b)
    return similarity

def search_similar_context(input_word, model, vocabs, word2index, top_n=10):
    word_similarities = {}

    try:
        # Get embedding for the input word
        input_embed = get_embed(model, input_word)

        # Ensure input_embed is a NumPy array
        if isinstance(input_embed, tuple):
            input_embed = np.array(input_embed)

        # Loop through each word in the vocabulary
        for word in vocabs:
            if word in word2index:  # Check if the word is in the vocabulary
                word_embed = get_embed(model, word)

                # Ensure word_embed is a NumPy array
                if isinstance(word_embed, tuple):
                    word_embed = np.array(word_embed)

                # Compute cosine similarity
                similarity = cosine_similarity(input_embed, word_embed)

                # Store similarity in the dictionary
                word_similarities[word] = similarity

        # Sort words by similarity in descending order
        sorted_similarities = sorted(word_similarities.items(), key=lambda x: x[1], reverse=True)

        # Remove the input word from the results
        filtered_words = [word for word, score in sorted_similarities if word != input_word]

        # Return the top N most similar words (excluding the input word)
        return filtered_words[:top_n]

    except KeyError:
        return "This word is not in the vocabulary. Please try a new word."
    except Exception as e:
        return f"An error occurred: {str(e)}"


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        input_data = request.get_json()
        input_word = input_data.get("search", "")
        similar_contexts = search_similar_context(input_word, model, vocabs, word2index)
        print(similar_contexts)
        return jsonify(similar_contexts)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
