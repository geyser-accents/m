from concurrent.futures import ThreadPoolExecutor, as_completed

max_workers = 2
batch_size = 20  

def insert_documents(documents, token_limit=8000):
    success_docs = []
    success_metadatas = []
    success_embeddings = []
    failed_documents = []
    failed_metadatas = []

    try:
        vector_store = chromadb.PersistentClient()
        collection = vector_store.get_or_create_collection("irm_collection98")
        tokenizer = tiktoken.encoding_for_model("gpt-4")

        def chunk_content(content):
            tokens = tokenizer.encode(content)
            chunks = [
                tokenizer.decode(tokens[i:i + token_limit])
                for i in range(0, len(tokens), token_limit)
            ]
            return chunks

        existing_documents = collection.get()['documents'] if collection.count() > 0 else set()
        print("Existing documents count:", len(existing_documents))

        all_contents = []
        all_metadatas = []

        for doc in documents:
            content = doc.page_content
            metadata = doc.metadata

            if content in existing_documents:
                print(f"Document already exist in collection. Skipping duplicate content: {content[:30]}...")
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

        def generate_embedding(content, metadata):
            try:
                embedding = get_embedding(content)
                time.sleep(1)
                return content, metadata, embedding
            except Exception as e:
                print(f"Error generating embedding for content: {content[:30]} - {e}")
                return content, metadata, None

        def add_to_collection(batch_docs, batch_metadatas, batch_embeddings):
            collection.add(
                documents=batch_docs,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                ids=[str(uuid4()) for _ in batch_docs]
            )
            print(f"Added {len(batch_docs)} documents to the collection.")

        futures = []
        with ThreadPoolExecutor(max_workers) as executor:
            for content, metadata in zip(all_contents, all_metadatas):
                futures.append(executor.submit(generate_embedding, content, metadata))

            current_batch_docs = []
            current_batch_metadatas = []
            current_batch_embeddings = []

            for future in tqdm(as_completed(futures), desc="Generating embeddings", total=len(all_contents)):
                content, metadata, embedding = future.result()
                if embedding is not None:
                    current_batch_docs.append(content)
                    current_batch_metadatas.append(metadata)
                    current_batch_embeddings.append(embedding)

                    if len(current_batch_docs) >= batch_size:
                        add_to_collection(current_batch_docs, current_batch_metadatas, current_batch_embeddings)
                        success_docs.extend(current_batch_docs)
                        success_metadatas.extend(current_batch_metadatas)
                        success_embeddings.extend(current_batch_embeddings)

                        current_batch_docs = []
                        current_batch_metadatas = []
                        current_batch_embeddings = []
                else:
                    failed_documents.append(content)
                    failed_metadatas.append(metadata)

            if current_batch_docs:
                add_to_collection(current_batch_docs, current_batch_metadatas, current_batch_embeddings)
                success_docs.extend(current_batch_docs)
                success_metadatas.extend(current_batch_metadatas)
                success_embeddings.extend(current_batch_embeddings)

        print(f"Successfully added {len(success_docs)} embeddings to the collection.")
        print(f"Failed to generate embeddings for {len(failed_documents)} document(s).")
        return success_docs, success_metadatas, failed_documents, failed_metadatas, success_embeddings
    except Exception as e:
        print(f"Error inserting documents: {e}")
        return success_docs, success_metadatas, failed_documents, failed_metadatas, success_embeddings

success_docs, success_metadatas, failed_documents, failed_metadatas, success_embeddings = insert_documents(all_documents)
