#imports
import os
import fitz
import re
from sentence_transformers import SentenceTransformer
import math
from typing import List, Tuple
import pandas as pd
from pacmap import PaCMAP
import numpy as np
import matplotlib.pyplot as plt 
from transformers import T5ForConditionalGeneration, T5Tokenizer
#parameters
data_directory = "Data/"
chunk_size = 250
chunk_overlap = 0
#reading files
def read_txt(file_path): #reading .txt files
    with open(file_path, 'r') as f:
        content = f.read()
        f.close()
    return content


def read_pdf(file_path): #reading .pdf files
    document = fitz.open(file_path)
    text = ""

    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

def read_file(file_path): #one function to read all supported file types
    _, file_extention = os.path.splitext(file_path)

    if file_extention.lower() == '.txt':
        return read_txt(file_path=file_path)
    elif file_extention.lower() == '.pdf':
        return read_pdf(file_path=file_path)
    else:
        return "Unsupported file type"
#storing the data as a corpus
def store_data(data_dir):
    data = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            data.append([file, read_file(os.path.join(root, file))])
    return data


def chunk_data(data, max_chunk_length=500):
    print("Chunking the data...")
    chunks = []
    for file in data:
        content = file[1]
        pdf_length = len(content)
        for i in range(pdf_length//max_chunk_length + 5):
            if content == "":
                break
            contentL = len(content)
            temp_length = min(max_chunk_length, contentL)
            temp_chunk = content[:temp_length]

            dotIndex = temp_chunk.rfind(". ")
            dotIndex2 = temp_chunk.rfind(".\n")
            qIndex = temp_chunk.rfind("? ")
            qIndex2 = temp_chunk.rfind("?\n")
            excIndex = temp_chunk.rfind("! ")
            excIndex2 = temp_chunk.rfind("!\n")
            entIndex = temp_chunk.rfind("\n")

            lastIndex = max(dotIndex, dotIndex2, qIndex, qIndex2, excIndex, excIndex2, entIndex)
            chunk = [file[0], content[:lastIndex]]
            content = content.replace(chunk[1], "", 1)
            chunks.append(chunk)
    return chunks

#embedding the chunks
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(chunks):
    print("Generating embeddings...")
    texts = [chunk[1] for chunk in chunks]
    chunk_embeddings = model.encode(texts)
    return list(zip([chunk[0] for chunk in chunks], chunk_embeddings))
    
#storing the embeddings in a vector DB
class VectorDB:
    def __init__(self):
        self.embeddings = []
        self.metadata = []

    def add_embeddings(self, embeddings: List[Tuple[str, List[float]]]):
        for source, embedding in embeddings:
            self.embeddings.append(embedding)
            self.metadata.append({"source": source})

    #using Euclidian distance for similarity search
    def _euclidean_distance(self, v1, v2):
        score = 0
        for i in range(len(v1)):
            score += (v1[i] - v2[i])**2
        return score * (1 / len(v1))

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[float, dict]]:
        if not self.embeddings:
            return []

        distances = [self._euclidean_distance(query_embedding, emb) for emb in self.embeddings]
        
        # Sort distances and get top_k (smallest distances)
        sorted_results = sorted(enumerate(zip(distances, self.metadata)), key=lambda x: x[1][0])
        return [(distance, metadata, index) for index, (distance, metadata) in sorted_results[:top_k]]

    #for seeing how many embeddings are in the vector store
    def __len__(self):
        return len(self.embeddings)
    
    #saving data to a .csv file
    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            for embedding, metadata in zip(self.embeddings, self.metadata):
                embedding_str = ','.join(map(str, embedding))
                f.write(f"{metadata['source']},{embedding_str}\n")

    #loading data from a .csv file
    @classmethod
    def load(cls, filepath: str):
        vector_store = cls()
        with open(filepath, 'r') as f:
            for line in f:
                embedding_str = line.strip().split(',')
                source = embedding_str.pop(0)
                embedding = list(map(float, embedding_str))
                vector_store.embeddings.append(embedding)
                vector_store.metadata.append({"source": source})
        return vector_store

#main body of program

if __name__ == "__main__":
    vector_store = VectorDB()
    print("Reading Files...")
    data = store_data(data_dir=data_directory)
    chunked_data = chunk_data(data=data, max_chunk_length=chunk_size)

    if not(os.path.exists("vector_store.csv")):    
        embeddings = generate_embeddings(chunked_data)
        vector_store.add_embeddings(embeddings=embeddings)
        print(f"Number of embeddings stored: {len(vector_store)}")
        vector_store.save('vector_store.csv')

    else:
        shouldUpdate = input("Do you wish to update the DB? (y/n)")
        if shouldUpdate == 'y':
            embeddings = generate_embeddings(chunked_data)
            vector_store.add_embeddings(embeddings=embeddings)
            print(f"Number of embeddings stored: {len(vector_store)}")
            vector_store.save('vector_store.csv')

        else:
            vector_store = VectorDB.load('vector_store.csv')


    query = input('->')
    query_embedding = model.encode([query])[0]

    results = vector_store.search(query_embedding, top_k=3)

    print(results[0])

    for distance, metadata, chunk_num in results:
            print(f"Source: {metadata['source']}, Distance: {distance:.4f}\n")
            print('\n'.join(chunked_data[chunk_num-1] + chunked_data[chunk_num] + chunked_data[chunk_num+1]))
            print()