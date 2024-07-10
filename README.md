1. **What is RAG?**
RAG stands for Retrieval Augmented Generation. RAG helps LLMs in various ways such as:
-Fact-checking
-Prevention of hallucination
-Updating their existing knowledge with new information
-Preventing the LLM from leaking sensitive info

3. **Feeding data into RAG**
Since most documents we use are large, we can not feed them into the system in one go, thus we have to break it down into "chunks".
If the chunk length, the chunk won't be precise enough, or in other words, it may include information from different topics.
If the chunk length is too small, the LLM will lose context.
Therefore we have to find a balanced chunk length to improve the accuracy of the system.
Apart from chunk length, another important parameter is chunk overlap. It demonstrates the amount of 2 chunks that are common between them.
We use chunk overlapping to give each chunk more context.

4. **Embedding**
The next step in building a retriever is embedding the chunks. we can use models such as BGE M3, BERT or SBERT.
An embedding gives each chunk an N-dimensional vector, which represents its "meaning". In other words, the closer the 2 vectors are, the closer their
corresponding chunks are.
We can calculate the distance between two chunks using measures such as cosine similarity, dot product, or Euclidian distance.

5. **Putting it all together**
When the user asks a question from the LLM, the retriever retrieves the closest chunks related to that query and then adds them as context to the user query 
and feeds them into the LLM.

6. **RAG proj1**
The jupyter notebook provided with this file contains a simple RAG retriever. It uses SBERT to generate the embeddings and saves them in a vector database on 
the local machine. For similarity search, it uses the aforementioned Euclidian distance. When presented with a query, the model then generates a new vector embedding of the query
and then searches for the closest vectors to it. It then returns the closest k vectors and their respective chunks and metadata. These chunks can then be passed onto an LLM as context alongside
the user prompt for it to generate more accurate and sold responses.
