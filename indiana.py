import time
from uuid import uuid4
from tqdm import tqdm
import chromadb
import tiktoken

def insert_documents(documents, token_limit=8000):
    success_docs = []
    success_metadatas = []
    success_embeddings = []
    failed_documents = []
    failed_metadatas = []

    try:
        vector_store = chromadb.PersistentClient()
        collection = vector_store.get_or_create_collection("irm_collection")
        tokenizer = tiktoken.encoding_for_model("gpt-4")

        def chunk_content(content):
            tokens = tokenizer.encode(content)
            chunks = [
                tokenizer.decode(tokens[i:i + token_limit])
                for i in range(0, len(tokens), token_limit)
            ]
            return chunks

        existing_documents = collection.get()['documents'] if collection.count() > 0 else set()

        all_contents = []
        all_metadatas = []

        for doc in documents:
            content = doc.page_content
            metadata = doc.metadata

            if content in existing_documents:
                print(f"Skipping duplicate content: {content[:30]}...")
                continue

            token_count = len(tokenizer.encode(content))
            if token_count > token_limit:
                chunks = chunk_content(content)
                for chunk in chunks:
                    all_contents.append(chunk)
                    all_metadatas.append(metadata)
            else:
                all_contents.append(content)
                all_metadatas.append(metadata)

        print("Creating embeddings now")
        for content, metadata in tqdm(zip(all_contents, all_metadatas), desc="Generating embeddings", total=len(all_contents)):
            try:
                embedding = get_embedding(content)
                if embedding is not None:
                    success_docs.append(content)
                    success_metadatas.append(metadata)
                    success_embeddings.append(embedding)
                else:
                    failed_documents.append(content)
                    failed_metadatas.append(metadata)
            except Exception as e:
                failed_documents.append(content)
                failed_metadatas.append(metadata)
                print(f"Unexpected error for content: {content[:10]} - {e}")
            time.sleep(2) 

        if success_docs:
            collection.add(
                documents=success_docs,
                embeddings=success_embeddings,
                metadatas=success_metadatas,
                ids=[str(uuid4()) for _ in success_docs]
            )
            print(f"Successfully added {len(success_docs)} embeddings to the collection.")
        else:
            print("No embeddings were successfully created to add to the collection.")

        print(f"Failed to generate embeddings for {len(failed_documents)} document(s).")
        return success_docs, success_metadatas, failed_documents, failed_metadatas
    except Exception as e:
        print(f"Error inserting documents: {e}")
        return success_docs, success_metadatas, failed_documents, failed_metadatas

success_docs, success_metadatas, failed_documents, failed_metadatas=insert_documents(all_documents)
