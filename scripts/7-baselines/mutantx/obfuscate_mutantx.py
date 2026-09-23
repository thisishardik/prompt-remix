import argparse
import json
import random
import time
import math
import copy
from pathlib import Path
from tqdm import tqdm
import numpy as np

import gensim.models.keyedvectors as word2vec

from hiatus.data import authbench_paths, blog_authorship_paths, hrs_paths
from hiatus.evaluation.embedding_extractor import EmbeddingExtractor

from Document import Document
from Mutant import Mutant_X

def computeNeighbours(word, wordVectorModel, neighborsCount=5):
    word = str(word).lower()
    tim = 0
    try:
        start = time.time()
        neighbors = list(wordVectorModel.similar_by_word(word, topn=neighborsCount))
        end = time.time()
        tim = end - start
    except:
        return -1, tim

    updated_neigbours = []
    for neighbor in neighbors:
        if neighbor[1] > 0.75:
            updated_neigbours.append(neighbor[0])
    if not updated_neigbours:
        return -1, tim

    return updated_neigbours, tim

def getWordNeighboursDictionary(inputText, wordVectorModel, totalNeighbours, global_dict):
    word_neighbours = {}
    for word in inputText:
        if word in global_dict:
            if global_dict[word] != -1:
                word_neighbours[word] = global_dict[word]
            continue
            
        neighbors, tim = computeNeighbours(word, wordVectorModel, totalNeighbours)
        global_dict[word] = neighbors
        if neighbors != -1:
            word_neighbours[word] = neighbors
    return word_neighbours

def getLabelAndProbabilities(document, extractor, profile_emb, author_id):
    mut_emb = extractor.encode([document.documentText], show_progress=False)[0]
    confidence = np.dot(mut_emb, profile_emb) / (np.linalg.norm(mut_emb) * np.linalg.norm(profile_emb) + 1e-9)
    confidence = (confidence + 1) / 2
    
    document.documentAuthorProbabilites = {author_id: confidence}
    document.documentAuthor = author_id

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["authbench", "blog", "hrs"])
    parser.add_argument("--language", type=str, required=True)
    parser.add_argument("--genre", type=str, default=None)
    parser.add_argument("--sample-name", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output-suffix", type=str, default="mutantx")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sampled-data-filepath", type=str, default=None)
    
    parser.add_argument("--pop-size", "-l", default=10, type=int)
    parser.add_argument("--topK", "-k", default=10, type=int)
    parser.add_argument("--crossover", "-c", default=0.5, type=float)
    parser.add_argument("--gens", "-M", default=15, type=int)
    parser.add_argument("--alpha", "-a", default=0.85, type=float)
    parser.add_argument("--beta", "-b", default=0.15, type=float)
    parser.add_argument("--replacements", "-Z", default=0.05, type=float)
    parser.add_argument("--replacementsLimit", "-rl", default=0.20, type=float)
    parser.add_argument("--allowedNeighbours", "-an", default=5, type=int)

    args = parser.parse_args()

    if args.dataset == "authbench":
        paths = authbench_paths(language=args.language, genre=args.genre, pipeline_suffix=args.output_suffix)
    elif args.dataset == "blog":
        paths = blog_authorship_paths(sample_name=args.sample_name, pipeline_suffix=args.output_suffix)
    elif args.dataset == "hrs":
        paths = hrs_paths(genre=args.genre, language=args.language, pipeline_suffix=args.output_suffix)["TA3"]["data"]

    with open(paths["queries"], "r") as f:
        queries = [json.loads(line) for line in f]
    with open(paths["candidates"], "r") as f:
        candidates = [json.loads(line) for line in f]
        
    if args.sampled_data_filepath:
        with open(args.sampled_data_filepath, "r") as f:
            sampled_docs = json.load(f)
        sampled_doc_ids = {doc["documentID"] for doc in sampled_docs}
        queries = [q for q in queries if q.get("documentID") in sampled_doc_ids]
        
    if args.limit:
        queries = queries[:args.limit]

    def get_author(obj):
        key = next((k for k in ["authorSetIDs", "authorIDs", "authorID", "author"] if k in obj), None)
        if key is None:
            raise KeyError(f"Could not find author key in keys: {obj.keys()}")
        val = obj[key]
        return val[0] if isinstance(val, list) else val

    candidate_texts = [c["fullText"] for c in candidates]
    candidate_labels = [get_author(c) for c in candidates]
    
    print("Encoding candidates...")
    extractor = EmbeddingExtractor(args.model_name)
    cand_embs = extractor.encode(candidate_texts, batch_size=args.batch_size)
    
    author_profiles = {}
    for i, label in enumerate(candidate_labels):
        if label not in author_profiles:
            author_profiles[label] = []
        author_profiles[label].append(cand_embs[i])
    for label in author_profiles:
        author_profiles[label] = np.mean(author_profiles[label], axis=0)

    print("Loading Word2Vec model for Mutant-X...")
    wordVectorModel = word2vec.KeyedVectors.load_word2vec_format("/gscratch/stf/hardiksr/pokerface/Word2vec.bin", binary=True)

    privatized_queries = []
    global_neighbor_dict = {}

    print(f"Running Mutant-X on {len(queries)} queries...")
    for q in tqdm(queries):
        author_id = get_author(q)
        if author_id not in author_profiles:
            profile_emb = np.zeros_like(cand_embs[0])
        else:
            profile_emb = author_profiles[author_id]

        originalDocument = Document(q["fullText"])
        getLabelAndProbabilities(originalDocument, extractor, profile_emb, author_id)

        neighboursDictionary = getWordNeighboursDictionary(originalDocument.documentWords, wordVectorModel, args.allowedNeighbours, global_neighbor_dict)
        mutant_x = Mutant_X(args.pop_size, args.topK, args.crossover, args.gens, neighboursDictionary, args.alpha, args.beta, args.replacements, args.replacementsLimit)

        indivisualsPopulation = [originalDocument]
        iteration_m = 1
        obfuscated = False

        while (iteration_m <= mutant_x.iterations) and (not obfuscated):
            generatedPopulation = []
            for indivisual in indivisualsPopulation:
                for i in range(0, mutant_x.generation):
                    indivisualCopy = copy.deepcopy(indivisual)
                    genDocument = mutant_x.makeReplacement(indivisualCopy)
                    getLabelAndProbabilities(genDocument, extractor, profile_emb, author_id)
                    generatedPopulation.append(genDocument)

            indivisualsPopulation.extend(generatedPopulation)

            if random.random() < mutant_x.crossover:
                choice1, choice2 = random.sample(indivisualsPopulation, 2)
                child_1, child_2 = mutant_x.single_point_crossover(copy.deepcopy(choice1), copy.deepcopy(choice2))
                getLabelAndProbabilities(child_1, extractor, profile_emb, author_id)
                getLabelAndProbabilities(child_2, extractor, profile_emb, author_id)
                indivisualsPopulation.extend([child_1, child_2])

            if originalDocument in indivisualsPopulation:
                indivisualsPopulation.remove(originalDocument)

            for indivisual in indivisualsPopulation:
                mutant_x.calculateFitness(originalDocument, indivisual)

            indivisualsPopulation.sort(key=lambda x: getattr(x, 'fitnessScore', 0.0), reverse=True)
            indivisualsPopulation = indivisualsPopulation[:mutant_x.topK]
            iteration_m += 1

        best_mutant = indivisualsPopulation[0] if len(indivisualsPopulation) > 0 else originalDocument
        
        q_copy = q.copy()
        q_copy["fullText"] = best_mutant.documentText
        privatized_queries.append(q_copy)

    output_path = paths["privatized"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for q in privatized_queries:
            f.write(json.dumps(q) + "\n")
            
    print(f"Privatized queries saved to {output_path}")

if __name__ == "__main__":
    main()
