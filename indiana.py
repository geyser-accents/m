import requests
from bs4 import BeautifulSoup

def parse_div(div, level):
    node_data = {
        'level': level,
        'text': div.get_text(strip=True, separator=' ')
    }

    if level == 0:
        heading_span = div.find('span', class_='heading')
        node_data['heading'] = heading_span.get_text(strip=True) if heading_span else ''

        child_divs = div.find_all('div', recursive=False)
        children = []
        for child_div in child_divs:
            child_data = parse_div(child_div, level + 1)
            children.append(child_data)

        if children:
            node_data['children'] = children

    return node_data


def remove_duplicate_text(node):
    if node['level'] == 0 and 'children' in node:
        if node['children']:
            first_child_text = node['children'][0]['text']


            start_index = node['text'].find(first_child_text)

            if start_index != -1:

                node['text'] = node['text'][:start_index].strip()


    if 'children' in node:
        for child in node['children']:
            remove_duplicate_text(child)

def scrape_uscode_section(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    section_div = soup.find('div', class_='section')
    if not section_div:
        print("No section found.")
        return

    section_data = {
        'url': url,
        'child': []
    }

    level0_divs = section_div.find_all('div', class_='subsection', recursive=False)

    for level0_div in level0_divs:
        level0_data = parse_div(level0_div, 0)
        remove_duplicate_text(level0_data)
        section_data['child'].append(level0_data)

    return section_data

url = "https://www.law.cornell.edu/uscode/text/26/1"
structured_data = scrape_uscode_section(url)



import requests
from bs4 import BeautifulSoup

def scrape(url, level=0):

    base_url = "https://www.law.cornell.edu"


    print(url)
    response = requests.get(url)


    if response.status_code != 200:
        print(f"Failed to retrieve {url}")
        return []


    soup = BeautifulSoup(response.content, 'html.parser')


    ol_element = soup.find('ol', class_='list-unstyled')


    if not ol_element:
        return []


    result = []


    for li in ol_element.find_all('li', class_='tocitem'):
        a_tag = li.find('a')
        if a_tag:

            title = a_tag.get_text(strip=True)
            relative_url = a_tag['href']
            full_url = base_url + relative_url


            item = {
                "title": title,
                "url": full_url,
                "level": level,
                "child": []
            }
            result.append(item)


            child_items = scrape(full_url, level + 1)
            item["child"].extend(child_items)

    return result

url = "https://www.law.cornell.edu/uscode/text/26"
scraped_data = scrape(url)

golden=scraped_data.copy()

scraped_data=golden

scraped_data[0]

import json

def post_process(scraped_data):

    level_names = {
        0: "Subtitle",
        1: "CHAPTER",
        2: "Subchapter",
        3: "PART",
        4: "Subpart",
        5: "Subpart's CHILD"
    }

    def process_node(node):

        node['level_name'] = level_names.get(node['level'], "")


        if 'level' in node:
            del node['level']


        if 'child' in node and len(node['child']) == 0:
            node['text_node'] = True
            del node['child']
        else:
            node['text_node'] = False


        for child in node.get('child', []):
            process_node(child)


    for item in scraped_data:
        process_node(item)


post_process(scraped_data)

with open("results_structuressv1.json", "w") as outfile:
    json.dump(scraped_data, outfile,indent=4)

import requests
from bs4 import BeautifulSoup

def word_count(text):
    return len(text.split())

def aggregate_level1(children, level_0_text):
    aggregated_level1 = []

    for i, child in enumerate(children):
        current_length = word_count(child['text'])

        if i == 0 and level_0_text and word_count(level_0_text) < 100:
            child['text'] = f"{level_0_text} {child['text']}".strip()

        if current_length < 100 and i > 0:
            print("combined")
            previous_child = aggregated_level1[-1]
            previous_child['text'] = f"{previous_child['text']} {child['text']}".strip()
        else:
            aggregated_level1.append(child)

    return aggregated_level1

def aggregate_level2(children, level1_text):
    aggregated_level2 = []

    for i, child in enumerate(children):
        current_length = word_count(child['text'])

        if i == 0 and level1_text and word_count(level1_text) < 100:
            child['text'] = f"{level1_text} {child['text']}".strip()

        if current_length < 100 and i > 0:
            print("combined")
            previous_child = aggregated_level2[-1]
            previous_child['text'] = f"{previous_child['text']} {child['text']}".strip()
        else:
            aggregated_level2.append(child)

    return aggregated_level2

def parse_div(div, level):
    node_data = {
        'level': level,
        'text': div.get_text(strip=True, separator=' ')
    }

    if level == 0:
        heading_span = div.find('span', class_='heading')
        node_data['heading'] = heading_span.get_text(strip=True) if heading_span else ''

        child_divs = div.find_all('div', recursive=False)
        children = []
        for child_div in child_divs:
            child_data = parse_div(child_div, level + 1)

            if word_count(child_data['text']) > 500:
                level2_divs = child_div.find_all('div', class_='subparagraph')
                if level2_divs:
                    child_data['children'] = []
                    for sub_div in level2_divs:
                        level2_data = {
                            'level': level + 2,  # Level 2
                            'text': sub_div.get_text(strip=True, separator=' ')
                        }
                        child_data['children'].append(level2_data)

                bold_heading_span = child_div.find('span', class_='heading bold')
                if bold_heading_span:
                    child_data['heading'] = bold_heading_span.get_text(strip=True)

            children.append(child_data)

        aggregated_level1 = aggregate_level1(children, node_data['text'])

        node_data['children'] = aggregated_level1

    elif level == 1:
        if 'children' in node_data:
            level1_text = node_data['text']
            node_data['children'] = aggregate_level2(node_data['children'], level1_text)

    return node_data

def remove_duplicate_text(node):
    if node['level'] == 0 and 'children' in node:
        if node['children']:
            first_child_text = node['children'][0]['text']
            start_index = node['text'].find(first_child_text)
            if start_index != -1:
                node['text'] = node['text'][:start_index].strip()

    if node['level'] == 1 and 'children' in node:
        if node['children']:
            first_level2_child_text = node['children'][0]['text']
            start_index = node['text'].find(first_level2_child_text)
            if start_index != -1:
                node['text'] = node['text'][:start_index].strip()

    if 'children' in node:
        for child in node['children']:
            remove_duplicate_text(child)

def scrape_uscode_section(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    section_div = soup.find('div', class_='section')
    if not section_div:
        print("No section found.")
        return

    section_data = {
        'url': url,
        'child': []
    }

    level0_divs = section_div.find_all('div', class_='subsection', recursive=False)

    for level0_div in level0_divs:
        level0_data = parse_div(level0_div, 0)
        remove_duplicate_text(level0_data)  # Remove duplicate text as post-processing
        section_data['child'].append(level0_data)

    return section_data

# # Example usage
# url = "https://www.law.cornell.edu/uscode/text/26/1"
# structured_data = scrape_uscode_section(url)



with open("results_structuressv1.json", "r") as outfile:
    scraped_data = json.load(outfile)


import json

def add_content_to_nodes(scraped_data):

    def process_node(node):

        if node.get('text_node') is True:
            print(node["title"])
            print(node["level_name"])
            print("\n")
            content = scrape_uscode_section(node['url'])
            node['content'] = content

        for child in node.get('child', []):
            process_node(child)

    for item in scraped_data:
        process_node(item)

add_content_to_nodes(scraped_data)

output_file = 'scraped_datav3.json'
with open(output_file, 'w') as f:
    json.dump(scraped_data, f, indent=4)



with open("scraped_datav3.json", "r") as outfile:
    data = json.load(outfile)

import time
from tqdm import tqdm

node_id_counter = 0
text_node_counter = 0

def get_next_node_id():
    global node_id_counter
    node_id = node_id_counter
    node_id_counter += 1
    return node_id


langchain_documents = []
text_node_counter=0

from langchain.schema import Document


def insert_data_into_neo4j(node, parent=None):
    global node_id_counter, text_node_counter

    node_label = node.get('level_name', 'Node')
    if node_label == "Subpart's CHILD":
        node_label = "Subpart_descendant"

    node_id = get_next_node_id()

    try:
        params = {
            "part_title": node['title'].replace("'", ""),
            "node_id": node_id
        }

    except Exception as e:
        print(f"Error inserting node {node['title']}: {e}. Retrying...")
        insert_data_into_neo4j(node, parent)

    if node.get('text_node', False):
        content = node.get('content', {})

        if content is not None:
            child_content = content.get('child', [])

            if child_content is not None:
                for level0 in child_content:
                    insert_level_nodes( level0, node_label, node_id)

    for child in node.get('child', []):
        insert_data_into_neo4j(child, parent={"title": node['title'], "label": node_label, "id": node_id})


def insert_level_nodes(node, parent_label, parent_id):
    global node_id_counter, text_node_counter

    level_label = f"Level{node['level']}"
    text = node['text']

    word_count = len(text.split())
    additional_label = "text_node_title" if word_count > 40 else ""

    if additional_label:
        lab = level_label.replace("'", "")
        label_string = f"{lab}"
    else:
        label_string = level_label.replace("'", "")

    node_id = get_next_node_id()

    try:
        if node.get('heading', ''):

            text_node_counter += 1
            print(text_node_counter)

            node_metadata = {"node_id": node_id,"Label":label_string,"Heading":node['heading']}
            parent_metadata = {"parent_label": parent_label, "parent_id": parent_id}
            combined_metadata = {**parent_metadata, **node_metadata}

            document = Document(
                page_content=text,
                metadata=combined_metadata
            )
            langchain_documents.append(document)
        else:

            text_node_counter += 1
            print(text_node_counter)

            node_metadata = {"node_id": node_id,"Label":label_string}
            parent_metadata = {"parent_label": parent_label, "parent_id": parent_id}
            combined_metadata = {**parent_metadata, **node_metadata}

            document = Document(
                page_content=text,
                metadata=combined_metadata
            )
            langchain_documents.append(document)

    except Exception as e:
        print(f"Error inserting level node: {e}. Retrying...")
        insert_level_nodes(node, parent_label, parent_id)

    if 'children' in node:
        for child in node['children']:
            insert_level_nodes(child, label_string, node_id)


for item in tqdm(data, desc="Inserting into Neo4j"):
    insert_data_into_neo4j(item)

!pip install openai
!pip install langchain-community langchain-core langchain-openai
!pip install -qU "langchain-chroma>=0.1.2"

def get_embedding(text):
  return [1,2,3]

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
        collection = vector_store.get_or_create_collection("uscode26_collection")
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

success_docs, success_metadatas, failed_documents, failed_metadatas=insert_documents(langchain_documents)

import pandas as pd
import time

import openai

from openai import OpenAI
client = OpenAI()

def get_embedding(text):
    response = client.embeddings.create(
    input=text,
    model="text-embedding-3-small"
)
    return response.data[0].embedding

import chromadb

persistent_client = chromadb.PersistentClient()
collection = persistent_client.get_or_create_collection("uscode26_collection")


query_text = "IRS mission"

query_embedding = get_embedding(query_text)

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)

documents = results['documents'][0]
metadatas = results['metadatas'][0]

top_docs=[]
for document, metadata in zip(documents, metadatas):
    top_docs.append({'content': document, 'metadata': metadata})

top_docs
