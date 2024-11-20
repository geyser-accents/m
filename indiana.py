Tax AgentThe agent operates exclusively using data from stored in a GraphDB, structured as a Knowledge Graph (KG).  When a user submits a query, the agent determines whether it is a new query or a follow-up to a previous question.

For a new query, the agent uses the tool, using the user query as input. This tool performs two  key tasks:

1. Document Retrieval

It retrieves the most semantically similar documents from the GraphDB. The retrieval process employs a specialized retrieval query designed to fetch not only the most relevant nodes but also data from their parent and child nodes. This approach ensures a comprehensive context by incorporating additional information that may enhance the response. Distinctions between a node’s data and that of its parent and child nodes are carefully maintained.


1. Answer Formulation

Using a prompt, the tool generates a response to the user query based on the retrieved documents. The emphasis is primarily on the node's data, while the parent and child nodes’ information is used as needed to enrich the response.

The tool produces two outputs:
● A direct answer to the user query.

● The set of retrieved documents.

The answer is presented to the user via the frontend, while the retrieved documents are cached for potential use in future interactions. To maintain focus and prevent confusion, any previously cached documents are cleared upon caching the new set.

If the query is identified as a follow-up or if the previously retrieved documents are sufficient to address the new question, the agent bypasses the retrieval step. Instead, it utilizes the cached documents to construct the response, avoiding redundant queries and ensuring operational efficiency.

By distinguishing between new and follow-up queries, the agent demonstrates intelligent decision-making and adaptability. The use of cached documents allows it to maintain continuity and context across interactions, ensuring responses are accurate, relevant, and comprehensive without unnecessarily querying the KG.

The tools used in IRM/ Revenue Bulletin/Title 26 works on the same logic as above.  In all three output from tool is answer to user query and retrieved documents. However, the retrieval query and KG used in each is different. This slightly alters the data retrieved in each case as retrieval query partially depends on structure of KG. 
