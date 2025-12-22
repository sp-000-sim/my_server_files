from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.schema.retriever import BaseRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.query_constructors.milvus import MilvusTranslator
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.types import StreamWriter
from langchain.chains.query_constructor.base import (
    load_query_constructor_runnable,
)
from custom_components.SelfQueryRetriever.matquest_config import Config
from custom_components.SelfQueryRetriever.patent_document_config import (
    PatentConfig,
)
from custom_components.SelfQueryRetriever.literature_document_config import (
    LiteratureConfig,
)
from custom_components.SelfQueryRetriever.tds_document_config import (
    TDSConfig,
)
from custom_components.DocumentListing import (
    ListingCoordinator,
    ListingContextManager,
    ListingOutputFormatter,
    get_document_type_name
)

from custom_components.PromptTemplates.document_listing_prompts import (
    DocumentListingPrompts
)
from custom_components.EntityMatching.create_filter import (
    expand_milvus_filter_through_attribute_matching,
)
from custom_components.timing_utils import timed_step
from custom_components.SelfQueryRetriever.unique_attribute_values import (
    get_unique_attribute_values,
)
from custom_components.PromptTemplates import (
    routing_prompts,
    generation_prompts,
    grader_prompts,
    query_handling_prompts,
    topic_classification_prompts,
)
from custom_components.token_tracking import (
    make_log_usage_node,
)
from concurrent.futures import ThreadPoolExecutor
from typing import TypedDict, Annotated, Sequence, List, Union, Tuple, Any, Optional, Dict, Callable
from datetime import datetime
import streamlit as st
import ast
import time

# Initialize URLs and API keys from Streamlit secrets
url = st.secrets["model_endpoints"]["matquest_endpoint"]
db_url = st.secrets["Vector_db"]["db_url"]

# Initialize models with authentication
mdl = Config.OLLAMA_MODELS[Config.ANSWER_MODEL_NAME_SHORTHAND]
gpt_oss_mdl = Config.GPT_OSS_ANSWER_MODEL_NAME
key = st.secrets["model_authentication"]["encoded_key"]
auth_headers = {"Authorization": f"Basic {key}"}

gen_model = ChatOllama(
    base_url=url,
    model=gpt_oss_mdl,
    temperature=0.0,
    num_ctx=20000,
    reasoning=True,
    client_kwargs={"headers": auth_headers},
)
# gen_model_json = ChatOllama(
#     base_url=url,
#     model=gpt_oss_mdl,
#     temperature=0.0,
#     num_ctx=20000,
#     reasoning=True,
#     client_kwargs={"headers": auth_headers},
#     tags=["json"],
# )
gen_model_json = ChatOllama(
    base_url=url,
    model=mdl,
    temperature=0.0,
    num_ctx=20000,
    format="json",
    client_kwargs={"headers": auth_headers},
    tags=["json"],
)
question_model = ChatOllama(
    base_url=url,
    model=gpt_oss_mdl,
    temperature=0.2,
    num_ctx=8000,
    reasoning=True,
    client_kwargs={"headers": auth_headers},
)


def get_embeddings_function(model_shortname: str) -> OllamaEmbeddings:
    """Create embeddings using specified model shortname.

    What this function does:
        Creates and returns an instance of OllamaEmbeddings configured with the
        appropriate model based on the provided shortname.

    Variables:
        - model_shortname: Short identifier for the embedding model to use.
        - embd_model: Dictionary mapping from model shortnames to their full model identifiers.
        - key: Authentication key for the Ollama API.
        - url: Base URL for the Ollama API endpoint.

    Logic:
        1. Defines a mapping between shortnames and full model identifiers
        2. Validates that the provided shortname exists in the mapping
        3. Creates and returns an OllamaEmbeddings object configured with:
           - The appropriate base URL for the API
           - The full model identifier corresponding to the shortname
           - Authentication headers
           - Configuration for progress display and temperature

    Returns:
        OllamaEmbeddings: A configured embedding model that can be used for
                          generating vector embeddings of text.
    """
    embd_model = {
        "granite": "granite-embedding:278m",
        "nomic": "nomic-embed-text:latest",
        "mxbai": "mxbai-embed-large:latest",
        "allmini": "all-minilm:v2",
    }

    if model_shortname not in embd_model:
        raise ValueError(
            f"Invalid model shortname. Must be one of {list(embd_model.keys())}"
        )

    return OllamaEmbeddings(
        model=embd_model[model_shortname],
        base_url=url,
        temperature=0,
        client_kwargs={"headers": auth_headers},
    )


# Configuration for collections
# COLLECTIONS_CONFIG = {
#     "patents": {
#         "collection_name": Config.PATENTS_COLLECTION_NAME,
#         "classifications": {
#             "Polymers": "Polymers",
#             "Glass": "Glass",
#             "Composites": "Composites",
#             "Unknown": "Unknown",
#         },
#     },
#     "literature": {
#         "collection_name": Config.LITERATURE_COLLECTION_NAME,
#         "classifications": {
#             "FRPolymers": "FRPolymers",
#             "Elastomers": "Elastomers",
#             "Composites": "Composites",
#             "Unknown": "Unknown",
#         },
#     },
#     "tds": {
#         "collection_name": Config.TDS_COLLECTION_NAME,
#         "classifications": {
#             "Generic": "Generic",  # TDS has only one general classification
#         },
#     },
# }


def COLLECTIONS_CONFIG():
    """Get current collections configuration with dynamic collection names."""
    return {
        "patents": {
            "collection_name": Config.get_dynamic_collection_name("patents"),
            "classifications": {
                "Polymers": "Polymers",
                "Glass": "Glass",
                "Composites": "Composites",
                "Unknown": "Unknown",
            },
        },
        "literature": {
            "collection_name": Config.get_dynamic_collection_name("literature"),
            "classifications": {
                "FRPolymers": "FRPolymers",
                "Elastomers": "Elastomers",
                "Composites": "Composites",
                "Unknown": "Unknown",
            },
        },
        "tds": {
            "collection_name": Config.get_dynamic_collection_name("tds"),
            "classifications": {
                "Generic": "Generic",
            },
        },
        "knowledge_base": {
            "collection_name": Config.get_dynamic_collection_name("knowledge_base"),
            "classifications": {
                "Generic": "Generic",
            },
        },
    }


def get_collection_name(collection_name):
    """Get the full collection name, potentially including the embedding model.

    What this function does:
        Returns the collection name, potentially modified to include information
        about the embedding model being used.

    Variables:
        - collection_name: Base name of the collection.

    Logic:
        Currently returns the collection name unchanged, but is structured to allow
        future expansion where the collection name might be modified to include
        the embedding model used.

    Returns:
        str: The (potentially modified) collection name to use with the database.
    """
    # TODO: Name the collection based on the embedding model used
    # return collection_name + "_" + embedding_model_shortname
    return collection_name


def create_milvus_db(collection_type: str) -> Milvus:
    """Create a Milvus database instance with the appropriate embedding model.

    What this function does:
        Creates and configures a Milvus vector database connection for a specific
        collection type, using the appropriate embedding model.

    Variables:
        - collection_type: The type of collection to access ('patents', 'literature', or 'tds').
        - collection_name: The name of the collection in the Milvus database.
        - COLLECTION_EMBEDDINGS: Mapping from collection types to embedding model shortnames.
        - model_shortname: The shortname of the embedding model to use for this collection.
        - embedding_function: The configured embedding function for the collection.

    Logic:
        1. Gets the correct collection name from the configuration
        2. Determines which embedding model to use based on the collection type
        3. Creates the appropriate embedding function using get_embeddings_function()
        4. Creates and returns a Milvus database instance with:
           - The determined embedding function
           - Connection arguments for the Milvus server
           - The correct collection name
           - Dynamic field support enabled

    Returns:
        Milvus: A configured Milvus database instance ready for vector searches.
    """
    collection_name = COLLECTIONS_CONFIG()[collection_type]["collection_name"]

    # Use nomic embeddings for TDS and knowledge base and granite for patents and literature
    # Map collection types to their embedding model shortnames
    COLLECTION_EMBEDDINGS = {
        "tds": "nomic",
        "patents": "granite",
        "literature": "granite",
        "knowledge_base": "nomic",
    }

    model_shortname = COLLECTION_EMBEDDINGS.get(
        collection_type, "granite"
    )  # Default to granite if collection type not found
    embedding_function = get_embeddings_function(model_shortname)

    return Milvus(
        embedding_function=embedding_function,
        vector_field=["dense", "sparse"],
        builtin_function=BM25BuiltInFunction(),
        connection_args={"uri": db_url},
        collection_name=collection_name,
        enable_dynamic_field=True,  # NOTE: This should be kept True if the collection being searched was created with dynamic fields
        consistency_level="Strong",
    )


def create_multi_query_retriever(
    db: Milvus,
    collection_type: str,
    data_source: str,
    k: int = Config.N_BASERETRIEVER_RETREIVED_DOCUMENTS,
) -> MultiQueryRetriever:
    """Create a multi-query retriever from a database with filtering by manual classification.

    What this function does:
        Creates a retriever that generates multiple query variations to improve
        retrieval performance, with filtering based on manual classification.

    Variables:
        - db: Milvus database instance to query against.
        - collection_type: Type of collection being queried ('patents', 'literature', or 'tds').
        - data_source: Manual classification value to filter documents by.
        - k: Number of documents to retrieve for each query.

    Logic:
        1. Validates that the provided data source is valid for the specified collection type
        2. Creates a MultiQueryRetriever that:
           - Includes the original query in the set of queries
           - Uses the provided Milvus database as the base retriever
           - Configures search parameters including:
             * Number of results to return (k)
             * Efficiency parameter for HNSW index (ef)
             * Filter expression to limit results to the specified classification
           - Uses question_model to generate query variations

    Returns:
        MultiQueryRetriever: A configured retriever that generates multiple query
                           variations for improved document retrieval.
    """
    # Verify the data source is valid for the collection type
    if data_source not in COLLECTIONS_CONFIG()[collection_type]["classifications"]:
        raise ValueError(f"Invalid data source {data_source} for {collection_type}")

    fetch_k = k * 5
    exploration_factor = fetch_k + 1
    return MultiQueryRetriever.from_llm(
        include_original=True,
        retriever=db.as_retriever(
            search_kwargs={
                "k": k,
                "fetch_k": fetch_k,
                "param": [
                    {"ef": exploration_factor},
                    {},
                ],
                "expr": f'manual_classification == "{COLLECTIONS_CONFIG()[collection_type]["classifications"][data_source]}"',
            }
        ),
        llm=question_model,
    )


@timed_step("create_milvus_filter", tag="filter_generation")
def create_milvus_filter(
    state: dict,
    collection_type: str,
    query: str,
    llm=None,
) -> Tuple[str, dict]:
    """Create a Milvus filter string based on the query and collection type.

    What this function does:
        Generates structured query filters for Milvus based on natural language queries,
        customized for different collection types.

    Variables:
        - collection_type: Type of collection ('patents', 'literature', or 'tds').
        - query: User's natural language query.
        - llm: Optional LLM model to use for query construction (created if not provided).
        - query_model_name: Name of the model to use for query construction.
        - attribute_info: Information about document attributes for the query constructor.
        - doc_description: Description of the document type for the query constructor.
        - examples: Example queries and filters for the query constructor.
        - query_constructor: Runnable chain for constructing structured queries.
        - structured_query: Generated structured query.
        - milvus_translator: Translator for converting structured queries to Milvus format.

    Logic:
        1. If no LLM is provided, creates one using the query-specific model from config
        2. Gets the appropriate configuration based on collection type:
           - Different configurations for patents, literature, and TDS collections
        3. Creates a query constructor chain with the configuration
        4. Generates a structured query from the natural language query
        5. If the structured query contains a filter:
           - Translates it to a Milvus-compatible format
           - Returns both the structured filter and translated Milvus filter
        6. If no filter or an error occurs, returns None for both filters

    Returns:
        Tuple[str, dict]: A tuple containing:
            - The Langchain structured filter expression or None
            - The translated Milvus filter expression or None
    """
    # Use structured query model for filter generation
    # start = time.perf_counter()
    if llm is None:
        # Initialize the model with query-specific model from config
        query_model_name = Config.OLLAMA_MODELS[Config.QUERY_MODEL_NAME_SHORTHAND]
        llm = ChatOllama(
            base_url=url,
            model=query_model_name,
            num_ctx=20000,
            temperature=0,
            client_kwargs={"headers": auth_headers},
        )

    # Get configuration based on collection type
    if collection_type == "patents":
        patent_config = PatentConfig()
        attribute_info = patent_config.attribute_info
        doc_description = patent_config.document_description
        examples = patent_config.examples
    elif collection_type == "literature":
        literature_config = LiteratureConfig()
        attribute_info = literature_config.attribute_info
        doc_description = literature_config.document_description
        examples = literature_config.examples
    elif collection_type == "tds":
        tds_config = TDSConfig()
        attribute_info = tds_config.attribute_info
        doc_description = tds_config.document_description
        examples = tds_config.examples
    else:
        raise NotImplementedError(
            "Filter generation for this collection type has not yet been implemented"
        )

    # Create query constructor chain
    query_constructor = load_query_constructor_runnable(
        llm,
        doc_description,
        attribute_info,
        examples=examples,
        fix_invalid=True,
    )
    # TODO: This runnable does not expose the usage token count. Need to figure out how to track it
    # The logging currently does nothing.
    query_construction_logger = make_log_usage_node("Structured Query Construction")
    query_construction_chain = query_constructor | query_construction_logger

    try:
        # Generate structured query
        structured_query = query_construction_chain.invoke(query)

        print(f"- Langchain Filter: {structured_query.filter}")
        # Get the Milvus translated query if there's a filter
        if structured_query.filter:
            milvus_translator = MilvusTranslator()
            _, translated_filter = milvus_translator.visit_structured_query(
                structured_query
            )
            print(f"- Translated Milvus Filter: {translated_filter}")

            collection_name = COLLECTIONS_CONFIG()[collection_type]["collection_name"]
            use_entity_matching_flag = True
            attributes_to_expand = []
            if collection_type == "literature":
                use_entity_matching_flag = Config.USE_ENTITY_MATCHING_FOR_LITERATURE
                attributes_to_expand = ["categories", "authors"]
            elif collection_type == "tds":
                use_entity_matching_flag = Config.USE_ENTITY_MATCHING_FOR_TDS
                attributes_to_expand = [
                    # "commercial_product_name", # TODO: Semantic similarity results in wrong documents also being included, only use substring matching
                    "general_compound_name",
                    "manufacturer_company_name",
                    "application",
                ]
            elif collection_type == "patents":
                use_entity_matching_flag = Config.USE_ENTITY_MATCHING_FOR_PATENTS
                attributes_to_expand = [
                    "classifications",
                    "assignee_name_orig",
                    "assignee_name_current",
                    "inventor_name",
                ]

            # For literature collection, expand categories with semantically similar ones
            if (
                translated_filter
                and "expr" in translated_filter
                and use_entity_matching_flag
            ):
                expanded_filter = expand_milvus_filter_through_attribute_matching(
                    translated_filter,
                    query,
                    attributes_to_expand,
                    collection_name,
                )
                print(f"- Expanded Milvus Filter: {expanded_filter}")
                return structured_query.filter, expanded_filter

            return structured_query.filter, translated_filter

        return None, None
    except Exception as e:
        print(f"Error generating Milvus filter: {e}")
        return None, None


def create_structured_retriever(
    state: dict,
    db: Milvus,
    collection_type: str,
    query: str,
) -> Union[MultiQueryRetriever, VectorStoreRetriever]:
    """Create a retriever that uses structured filters for search.

    What this function does:
        Creates a retriever that combines vector search with structured filters
        generated from natural language queries.

    Variables:
        - db: Milvus database instance to query against.
        - collection_type: Type of collection ('patents', 'literature', or 'tds').
        - query: User's natural language query.
        - structured_filter: Generated Langchain structured filter.
        - translated_filter: Milvus-compatible version of the structured filter.
        - use_multi_query: Flag indicating whether to use MultiQueryRetriever.
        - search_kwargs: Parameters for the vector search.
        - question_llm: LLM for generating query variations in MultiQueryRetriever.

    Logic:
        1. Generates structured filters based on the query and collection type
        2. Logs the generated filters
        3. Determines whether to use MultiQueryRetriever based on collection type and config
        4. Sets up base search parameters including:
           - Number of results to return
           - Efficiency parameter for HNSW index
        5. Adds the filter expression to search parameters if a filter was generated
        6. Returns either:
           - A MultiQueryRetriever if configured to use multiple queries, or
           - A simple VectorStoreRetriever otherwise

    Returns:
        Union[MultiQueryRetriever, VectorStoreRetriever]: A configured retriever that
        combines vector search with structured filters.
    """
    # Get structured filter if using self query
    structured_filter, translated_filter = create_milvus_filter(
        state, collection_type, query
    )  # Determine whether to use MultiQueryRetriever based on collection type
    use_multi_query = False
    if collection_type == "patents":
        use_multi_query = Config.USE_MULTIQUERYRETRIEVER_FOR_PATENTS
    elif collection_type == "literature":
        use_multi_query = Config.USE_MULTIQUERYRETRIEVER_FOR_LITERATURE
    elif collection_type == "tds":
        use_multi_query = Config.USE_MULTIQUERYRETRIEVER_FOR_TDS

    # Base search kwargs
    fetch_k = Config.N_SELFQUERYRETRIEVER_RETREIVED_DOCUMENTS * 5
    exploration_factor = fetch_k + 1
    search_kwargs = {
        "k": Config.N_SELFQUERYRETRIEVER_RETREIVED_DOCUMENTS,
        "fetch_k": fetch_k,
        "param": [
            {
                "ef": exploration_factor
            },  # for 'dense' # NOTE: ef parameter must be > k for HNSW index
            {},  # for 'sparse' (BM25 usually doesn't need params)
        ],
    }

    # Add expression to search kwargs if filter exists
    if translated_filter is not None:
        search_kwargs["expr"] = translated_filter["expr"]

    # Return appropriate retriever based on configuration
    if use_multi_query:
        # Use a different model for query generation in MultiQueryRetriever
        question_llm = ChatOllama(
            base_url=url,
            model=Config.OLLAMA_MODELS[Config.ANSWER_MODEL_NAME_SHORTHAND],
            temperature=0.2,
            num_ctx=8000,
            client_kwargs={"headers": auth_headers},
        )

        # TODO: Figure out a way to track token usage in this situation

        return MultiQueryRetriever.from_llm(
            retriever=db.as_retriever(search_kwargs=search_kwargs),
            llm=question_llm,
            include_original=True,
        )
    else:
        # Return simple vector store retriever
        print(f"Search kwargs: {search_kwargs}")
        return db.as_retriever(search_kwargs=search_kwargs)


def create_compressed_retriever(
    retriever: Union[MultiQueryRetriever, SelfQueryRetriever],
) -> ContextualCompressionRetriever:
    """Create a compressed retriever with flashrank reranking.

    What this function does:
        Wraps a base retriever with a reranking compressor to improve the
        relevance of retrieved documents.

    Variables:
        - retriever: Base retriever to wrap with compression.
        - compressor: FlashrankRerank instance for reranking documents.

    Logic:
        1. Creates a FlashrankRerank compressor with:
           - The specified reranking model from configuration
           - Configured number of top documents to keep
           - Similarity threshold for filtering results
        2. Creates and returns a ContextualCompressionRetriever that:
           - Uses the created compressor
           - Uses the provided base retriever

    Returns:
        ContextualCompressionRetriever: A retriever that enhances results using
                                       reranking compression.
    """
    compressor = FlashrankRerank(
        model=Config.FLASHRANK_MODEL,
        top_n=Config.N_TOP_RERANKED_DOCUMENTS,
        score_threshold=Config.RERANKING_SIMILARITY_THRESHOLD,
    )
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever,
    )


# # Initialize databases with single collection for each category
# databases = {
#     category: create_milvus_db(category) for category in COLLECTIONS_CONFIG.keys()
# }


# Initialize databases with single collection for each category
def _create_databases():
    """Create databases dictionary with current collection names."""
    return {
        category: create_milvus_db(category) for category in COLLECTIONS_CONFIG().keys()
    }


# Create initial databases
databases = _create_databases()


def refresh_databases():
    """Refresh databases with current collection configuration."""
    global databases
    print("🔄 Refreshing databases with new collection names...")
    databases = _create_databases()
    print("✅ Databases refreshed!")


def get_base_retriever(
    state: dict,
    collection_type: str,
    classification: str,
    query: str = None,
) -> Union[
    MultiQueryRetriever, SelfQueryRetriever, VectorStoreRetriever, BaseRetriever
]:
    """Get the appropriate base retriever based on collection type and configuration.

    What this function does:
        Creates the appropriate retriever based on configuration settings and
        collection type, with optional structured query support.

    Variables:
        - collection_type: Type of collection ('patents', 'literature', or 'tds').
        - classification: Classification type from COLLECTIONS_CONFIG.
        - query: Query for retrieval and/or generating structured filter.
        - use_multi_query: Flag indicating whether to use MultiQueryRetriever.

    Logic:
        1. Checks if self-query should be used (based on Config.USE_SELF_QUERY):
           - If enabled and a query is provided, creates a structured retriever
        2. If not using self-query, determines if MultiQueryRetriever should be used:
           - Checks configuration flag specific to the collection type
        3. Returns the appropriate retriever:
           - MultiQueryRetriever if configured for the collection type
           - Simple vector retriever otherwise, with filters for classification

    Returns:
        Union[MultiQueryRetriever, SelfQueryRetriever, VectorStoreRetriever, BaseRetriever]:
        The appropriate retriever instance based on configuration settings.
    """
    # First check if we should use self-query
    # NOTE: For knowledge base collection type, we do not use self-query since the metadata is not structured in the same way as other collections
    if Config.USE_SELF_QUERY and query and collection_type != "knowledge_base":
        return create_structured_retriever(
            state,
            db=databases[collection_type],
            collection_type=collection_type,
            query=query,
        )

    # If not using self-query, check if we should use MultiQueryRetriever based on collection type
    use_multi_query = False
    if collection_type == "patents":
        use_multi_query = Config.USE_MULTIQUERYRETRIEVER_FOR_PATENTS
    elif collection_type == "literature":
        use_multi_query = Config.USE_MULTIQUERYRETRIEVER_FOR_LITERATURE
    elif collection_type == "tds":
        use_multi_query = Config.USE_MULTIQUERYRETRIEVER_FOR_TDS
    elif collection_type == "knowledge_base":
        use_multi_query = Config.USE_MULTIQUERYRETRIEVER_FOR_KNOWLEDGE_BASE

    if use_multi_query:
        return create_multi_query_retriever(
            db=databases[collection_type],
            collection_type=collection_type,
            data_source=classification,
        )
    else:
        # Use simple retriever
        fetch_k = Config.N_BASERETRIEVER_RETREIVED_DOCUMENTS * 5
        exploration_factor = fetch_k + 1
        return databases[collection_type].as_retriever(
            search_kwargs={
                "k": Config.N_BASERETRIEVER_RETREIVED_DOCUMENTS,
                "fetch_k": fetch_k,
                "param": [
                    {"ef": exploration_factor},
                    {},
                ],
            }
        )


def get_retriever(
    state: dict,
    collection_type: str,
    classification: str,
    query: str,
    use_reranking: bool = Config.USE_RERANKING,
) -> Union[
    VectorStoreRetriever,
    MultiQueryRetriever,
    SelfQueryRetriever,
    ContextualCompressionRetriever,
]:
    """Get the appropriate retriever with optional reranking based on configuration.

    What this function does:
        Creates and returns a retriever, optionally enhanced with reranking,
        based on configuration settings.

    Variables:
        - collection_type: Type of collection ('patents', 'literature' or 'tds').
        - classification: Classification type from COLLECTIONS_CONFIG.
        - query: Query for generating structured filter.
        - use_reranking: Flag indicating whether to use reranking.
        - base_retriever: The base retriever before potential reranking.

    Logic:
        1. Gets the appropriate base retriever using get_base_retriever()
        2. If reranking is enabled:
           - Wraps the base retriever with a compressed retriever that uses reranking
           - Returns the enhanced retriever
        3. If reranking is disabled:
           - Returns the base retriever without modification

    Returns:
        Union[VectorStoreRetriever, MultiQueryRetriever, SelfQueryRetriever, ContextualCompressionRetriever]:
        The appropriate retriever, optionally enhanced with reranking.
    """
    base_retriever = get_base_retriever(state, collection_type, classification, query)

    if use_reranking:
        return create_compressed_retriever(base_retriever)
    return base_retriever


# TODO: Prompt template markers should be improved and placed appropriately
# Question Router Initialization
question_routing_llm = ChatOllama(
    base_url=url,
    model=mdl,
    format="json",
    temperature=0,
    client_kwargs={"headers": auth_headers},
)
# TODO: This sub-collection routing is no longer needed and should be removed in future
question_router_logger = make_log_usage_node("Question Sub-Collection Router")

# Prompt for routing to patent database
patent_routing_prompt = routing_prompts.patent_routing_prompt

# Chain for routing to patent database
patent_question_router = (
    patent_routing_prompt
    | question_routing_llm
    | question_router_logger
    | JsonOutputParser()
)

# Prompt for routing to literature database
literature_routing_prompt = routing_prompts.literature_routing_prompt

# Chain for routing to literature database
literature_question_router = (
    literature_routing_prompt
    | question_routing_llm
    | question_router_logger
    | JsonOutputParser()
)

# Prompt for routing to TDS database
tds_routing_prompt = routing_prompts.tds_routing_prompt

# Chain for routing to TDS database
tds_question_router = (
    tds_routing_prompt
    | question_routing_llm
    | question_router_logger
    | JsonOutputParser()
)

# Document Grading Components
retrieval_grader_llm = ChatOllama(
    base_url=url,
    model=mdl,
    format="json",
    temperature=0,
    client_kwargs={"headers": auth_headers},
)

# Prompt for grading retrieved documents
retrieval_grader_prompt = grader_prompts.retrieval_grader_prompt
retrieval_grader_logger = make_log_usage_node("Retrieval Grader")

# Chain for document grading
retrieval_grader = (
    retrieval_grader_prompt
    | retrieval_grader_llm
    | retrieval_grader_logger
    | JsonOutputParser()
)


def join_patent_documents(documents):
    """Join documents while preserving selected metadata and formatting context for patents.

    What this function does:
        Creates a formatted string that combines multiple patent document chunks,
        preserving important metadata fields in a consistent format.

    Variables:
        - documents: List of Document objects with page_content and metadata.
        - formatted_docs: List to store formatted document strings.
        - selected_metadata: Ordered list of metadata fields to include.
        - doc_text: Formatted text for each document chunk.
        - metadata_text: List of formatted metadata field-value pairs.
        - field_labels: Mapping from internal field names to human-readable labels.
        - joined_documents: Final combined string of all document chunks.

    Logic:
        1. Defines an ordered list of metadata fields to preserve
        2. Iterates through each document:
           - Creates a header with document chunk number
           - Adds the document content
           - Processes selected metadata fields in the specified order
           - Converts internal field names to more descriptive labels
           - Formats metadata as field-value pairs
           - Combines metadata with document content
        3. Joins all formatted document chunks with separators

    Returns:
        str: A formatted string containing all document chunks with their metadata,
             ready for use in RAG prompts.
    """
    formatted_docs = []
    selected_metadata = [
        "patent",  # Patent ID first
        "title",  # Title second
        "abstract_text",  # Abstract third
        "classification",  # Classification third
        "inventor_name",  # People/organizations next
        "assignee_name_orig",
        "assignee_name_current",
        "pub_date",  # Dates last
        "expiration_date",
        "url",
        "pk",  # Internal ID at the very end
    ]

    for i, doc in enumerate(documents, 1):
        doc_text = "\n---\nDocument:\n"
        doc_text += f"Content:\n{doc.page_content}\n"

        # Add only selected metadata in specified order
        metadata_text = []
        for field in selected_metadata:
            if (
                field in doc.metadata and doc.metadata[field]
            ):  # Only include if field exists and has value
                value = doc.metadata[field]
                # Map field names to more descriptive labels
                field_labels = {
                    "patent": "Patent ID",
                    "title": "Patent Title",
                    "abstract_text": "Abstract",
                    "classification": "Classification",
                    "inventor_name": "Inventors",
                    "assignee_name_orig": "Original Assignee",
                    "assignee_name_current": "Current Assignee",
                    "pub_date": "Publication Date",
                    "expiration_date": "Expiration Date",
                    "url": "Patent URL",
                    "pk": "Chunk ID (pk)",
                }
                field_name = field_labels.get(field, field.replace("_", " ").title())
                metadata_text.append(f"{field_name}: {value}")

        if metadata_text:
            doc_text += "Metadata:\n  " + "\n  ".join(metadata_text) + "\n"

        formatted_docs.append(doc_text)

    joined_documents = "\n---\n".join(formatted_docs)
    return joined_documents


def join_literature_documents(documents):
    """Join documents while preserving selected metadata and formatting context for literature.

    What this function does:
        Creates a formatted string that combines multiple literature document chunks,
        preserving important metadata fields in a consistent format.

    Variables:
        - documents: List of Document objects with page_content and metadata.
        - formatted_docs: List to store formatted document strings.
        - selected_metadata: Ordered list of metadata fields to include.
        - doc_text: Formatted text for each document chunk.
        - metadata_text: List of formatted metadata field-value pairs.
        - field_labels: Mapping from internal field names to human-readable labels.
        - joined_documents: Final combined string of all document chunks.

    Logic:
        1. Defines an ordered list of metadata fields to preserve
        2. Iterates through each document:
           - Creates a header with document chunk number
           - Adds the document content
           - Processes selected metadata fields in the specified order
           - Converts internal field names to more descriptive labels
           - Special handling for author lists
           - Formats metadata as field-value pairs
           - Combines metadata with document content
        3. Joins all formatted document chunks with separators

    Returns:
        str: A formatted string containing all document chunks with their metadata,
             ready for use in RAG prompts.
    """
    formatted_docs = []
    selected_metadata = [
        "doi",  # DOI first
        "title",  # Title second
        "abstract",  # Abstract third
        "authors",  # People next
        "categories",  # Categories next
        "published_time",  # Date information
        "submitted_time",  # Date information
        "pk",  # Internal ID at the very end
    ]

    for i, doc in enumerate(documents, 1):
        doc_text = "\n---\nDocument:\n"
        doc_text += f"Content:\n{doc.page_content}\n"

        # Add only selected metadata in specified order
        metadata_text = []
        for field in selected_metadata:
            if (
                field in doc.metadata and doc.metadata[field]
            ):  # Only include if field exists and has value
                value = doc.metadata[field]
                # Map field names to more descriptive labels
                field_labels = {
                    "doi": "DOI",
                    "title": "Paper Title",
                    "abstract": "Abstract",
                    "authors": "Authors",
                    "categories": "Categories",
                    "published_time": "Publication Date",
                    "submitted_time": "Submission Date",
                    "pk": "Chunk ID (pk)",
                }
                field_name = field_labels.get(field, field.replace("_", " ").title())

                # Special handling for author lists
                if field == "authors" and isinstance(value, list):
                    value = ", ".join(value)

                metadata_text.append(f"{field_name}: {value}")

        if metadata_text:
            doc_text += "Metadata:\n  " + "\n  ".join(metadata_text) + "\n"

        formatted_docs.append(doc_text)

    joined_documents = "\n---\n".join(formatted_docs)
    return joined_documents


def join_tds_documents(documents):
    """Join documents while preserving selected metadata and formatting context for technical data sheets.

    What this function does:
        Creates a formatted string that combines multiple technical data sheet document chunks,
        preserving important metadata fields in a consistent format.

    Variables:
        - documents: List of Document objects with page_content and metadata.
        - formatted_docs: List to store formatted document strings.
        - selected_metadata: Ordered list of metadata fields to include.
        - doc_text: Formatted text for each document chunk.
        - metadata_text: List of formatted metadata field-value pairs.
        - field_labels: Mapping from internal field names to human-readable labels.
        - joined_documents: Final combined string of all document chunks.

    Logic:
        1. Defines an ordered list of metadata fields to preserve
        2. Iterates through each document:
           - Creates a header with document chunk number
           - Adds the document content
           - Processes selected metadata fields in the specified order
           - Converts internal field names to more descriptive labels
           - Formats metadata as field-value pairs
           - Combines metadata with document content
        3. Joins all formatted document chunks with separators

    Returns:
        str: A formatted string containing all document chunks with their metadata,
             ready for use in RAG prompts.
    """
    formatted_docs = []
    selected_metadata = [
        "commercial_product_name",  # Product name first
        "general_compound_name",  # Chemical name second
        "manufacturer_company_name",  # Manufacturer third
        "application",  # Application fourth
        "year",  # Year fifth
        "pk",  # Internal ID at the very end
    ]

    for i, doc in enumerate(documents, 1):
        doc_text = "\n---\nDocument:\n"
        doc_text += f"Content:\n{doc.page_content}\n"

        # Add only selected metadata in specified order
        metadata_text = []
        for field in selected_metadata:
            if (
                field in doc.metadata and doc.metadata[field]
            ):  # Only include if field exists and has value
                value = doc.metadata[field]
                # Map field names to more descriptive labels
                field_labels = {
                    "commercial_product_name": "Product Name",
                    "general_compound_name": "General Compound Type",
                    "manufacturer_company_name": "Manufacturer",
                    "application": "Application",
                    "year": "Year",
                    "pk": "Chunk ID (pk)",
                }
                field_name = field_labels.get(field, field.replace("_", " ").title())
                metadata_text.append(f"{field_name}: {value}")

        if metadata_text:
            doc_text += "Metadata:\n  " + "\n  ".join(metadata_text) + "\n"

        formatted_docs.append(doc_text)

    joined_documents = "\n---\n".join(formatted_docs)
    return joined_documents


def join_knowledge_base_documents(documents):
    """Join documents while preserving selected metadata and formatting context for knowledge base documents.

    What this function does:
        Creates a formatted string that combines multiple knowledge base document chunks,
        preserving minimal but important metadata fields in a consistent format.

    Variables:
        - documents: List of Document objects with page_content and metadata.
        - formatted_docs: List to store formatted document strings.
        - selected_metadata: Ordered list of metadata fields to include.
        - doc_text: Formatted text for each document chunk.
        - metadata_text: List of formatted metadata field-value pairs.
        - field_labels: Mapping from internal field names to human-readable labels.
        - joined_documents: Final combined string of all document chunks.

    Logic:
        1. Defines a minimal list of metadata fields to preserve (title, url, chunk id)
        2. Iterates through each document:
           - Creates a header with document chunk number
           - Adds the document content
           - Processes selected metadata fields in the specified order
           - Converts internal field names to more descriptive labels
           - Formats metadata as field-value pairs
           - Combines metadata with document content
        3. Joins all formatted document chunks with separators

    Returns:
        str: A formatted string containing all document chunks with their metadata,
             ready for use in RAG prompts.
    """
    formatted_docs = []
    selected_metadata = [
        "title",  # Document title first
        "url",  # URL second (if available)
        "pk",  # Internal ID at the end
        "patent",
    ]

    for i, doc in enumerate(documents, 1):
        doc_text = "\n---\nDocument:\n"
        doc_text += f"Content:\n{doc.page_content}\n"

        # Add only selected metadata in specified order
        metadata_text = []
        for field in selected_metadata:
            if (
                field in doc.metadata and doc.metadata[field]
            ):  # Only include if field exists and has value
                value = doc.metadata[field]
                # Map field names to more descriptive labels
                field_labels = {
                    "title": "Document Title",
                    "url": "Document URL",
                    "pk": "Chunk ID (pk)",
                    "patent": "Patent Number",
                }
                field_name = field_labels.get(field, field.replace("_", " ").title())
                metadata_text.append(f"{field_name}: {value}")

        if metadata_text:
            doc_text += "Metadata:\n  " + "\n  ".join(metadata_text) + "\n"

        formatted_docs.append(doc_text)

    joined_documents = "\n---\n".join(formatted_docs)
    return joined_documents


def join_documents(documents):
    """Join document contents into a single string.

    What this function does:
        Combines the content of multiple documents into a single string.

    Variables:
        - documents: List of document contents to join.
        - joined_documents: The resulting combined string.

    Logic:
        1. Joins all document contents with spaces between them
        2. Logs the joined content for debugging
        3. Returns the joined string

    Returns:
        str: A single string containing all document contents joined with spaces.
    """
    joined_documents = " ".join(document for document in documents)
    print(f"Joined Documents: {joined_documents}")
    return " ".join(document for document in documents)


def get_sources(document_sources):
    """Combine document source identifiers into a single string.

    What this function does:
        Joins a list of document source identifiers into a single space-separated string.

    Variables:
        - document_sources: List of source identifiers.

    Logic:
        Joins all source identifiers with spaces between them.

    Returns:
        str: A single string containing all source identifiers joined with spaces.
    """
    return " ".join(source for source in document_sources)


# Prompt template for answer generation
answer_generation_prompt = generation_prompts.answer_generation_prompt
answer_generation_logger = make_log_usage_node("RAG Answer Generation without Sources")

# Standard RAG chain using join_documents utility
rag_chainA = (
    {
        "context": join_documents,
        "question": RunnablePassthrough(),
        "sources": get_sources,
    }
    | answer_generation_prompt
    | gen_model
    | answer_generation_logger
    | StrOutputParser()
)


# Prompt for JSON-formatted answer generation
# Dynamic prompt template selection based on model name
def get_json_answer_generation_prompt(model_name: str):
    """Select appropriate prompt template based on model name"""
    if "gpt" in model_name.lower():
        return generation_prompts.json_answer_generation_prompt_gpt
    else:
        return generation_prompts.json_answer_generation_prompt_llama


json_answer_generation_prompt = get_json_answer_generation_prompt(gen_model_json.model)
json_answer_generation_logger = make_log_usage_node(
    "RAG Answer Generation with Sources"
)


# Generate Patent Summary based on template
# Dynamic prompt template selection based on model name
def get_patent_summary_generation_prompt(model_name: str):
    """Select appropriate prompt template based on model name"""
    if "gpt" in model_name.lower():
        return generation_prompts.patent_summary_generation_prompt_gpt
    else:
        return generation_prompts.patent_summary_generation_prompt_llama


patent_summary_generation_prompt = get_patent_summary_generation_prompt(
    gen_model_json.model
)
patent_summary_generation_logger = make_log_usage_node("Patent Summary Generation")


def print_full_prompt(prompt):
    """Print the full prompt before generation for debugging.

    What this function does:
        Logs the full prompt that will be sent to the model for debugging purposes.

    Variables:
        - prompt: The complete prompt string to be logged and returned.

    Logic:
        1. Prints a header to mark the start of the prompt section
        2. Prints the full prompt content
        3. Prints a footer to mark the end of the prompt section
        4. Returns the original prompt unchanged

    Returns:
        str: The original prompt, unchanged (pass-through function).
    """
    print("\n=== FULL PROMPT ===")
    print(f"PROMPT:\n{prompt}")
    print("===================\n")
    return prompt


# Alternative RAG chains for different use cases
rag_chainB = (
    {
        "context": join_documents,
        "question": RunnablePassthrough(),
    }
    | json_answer_generation_prompt
    | gen_model_json
    | json_answer_generation_logger
    | JsonOutputParser()
)

rag_chainC = (
    json_answer_generation_prompt
    # | print_full_prompt
    | gen_model_json
    | json_answer_generation_logger
    | JsonOutputParser()
)

patent_summary_chain = (
    patent_summary_generation_prompt
    # | print_full_prompt
    | gen_model_json
    | patent_summary_generation_logger
    | JsonOutputParser()
)

# Ground truth grading components
ground_truth_grader_llm = ChatOllama(
    base_url=url,
    model=mdl,
    format="json",
    temperature=0.2,
    num_ctx=8096,
    client_kwargs={"headers": auth_headers},
)

# Prompt for ground truth grading
ground_truth_grader_prompt = grader_prompts.ground_truth_grader_prompt
ground_truth_grader_logger = make_log_usage_node("Ground Truth Grader")

# Chain for ground truth grading
ground_truth_grader = (
    ground_truth_grader_prompt
    | ground_truth_grader_llm
    | ground_truth_grader_logger
    | JsonOutputParser()
)

# Answer relevance grading components
answer_relevance_grader_llm = ChatOllama(
    base_url=url,
    model=mdl,
    format="json",
    temperature=0,
    client_kwargs={"headers": auth_headers},
)

# Prompt for answer relevance grading
answer_relevance_grader_prompt = grader_prompts.answer_relevance_grader_prompt
answer_relevance_grader_logger = make_log_usage_node("Answer Relevance Grader")

# Chain for answer relevance grading
answer_relevance_grader = (
    answer_relevance_grader_prompt
    | answer_relevance_grader_llm
    | answer_relevance_grader_logger
    | JsonOutputParser()
)

# Standalone query detection and reformulation
chat_model = ChatOllama(
    base_url=url,
    model=mdl,
    temperature=0,
    client_kwargs={"headers": auth_headers},
)

standalone_detection_llm = ChatOllama(
    base_url=url,
    model="deepseek-r1:8b",
    reasoning=True,
    format="json",
    temperature=0,
    num_ctx=10000,
    client_kwargs={"headers": auth_headers},
)

# Prompt for standalone query detection
standalone_detection_prompt = query_handling_prompts.standalone_detection_prompt
standalone_detection_logger = make_log_usage_node("Standalone Detection")

# Chain for standalone query detection
standalone_detection_chain = (
    standalone_detection_prompt
    | standalone_detection_llm
    | standalone_detection_logger
    | JsonOutputParser()
)

reformulator_llm = ChatOllama(
    base_url=url,
    model=gpt_oss_mdl,
    reasoning=True,
    temperature=0,
    num_ctx=15000,
    client_kwargs={"headers": auth_headers},
)


# Prompt for question reformulation
reformulation_prompt = query_handling_prompts.reformulation_prompt
reformulation_logger = make_log_usage_node("Query Reformulation")

# Chain for question reformulation
reformulation_chain = (
    reformulation_prompt | reformulator_llm | reformulation_logger | StrOutputParser()
)

# Chemistry question detection
chemistry_check_llm = ChatOllama(
    base_url=url,
    model=mdl,
    format="json",
    temperature=0,
    client_kwargs={"headers": auth_headers},
)

# Prompt for chemistry question checking
chemistry_checking_prompt = query_handling_prompts.chemistry_checking_prompt
chemistry_check_logger = make_log_usage_node("Chemistry Checker")

# Chain for chemistry query checking
chemistry_query_checker = (
    chemistry_checking_prompt
    | chemistry_check_llm
    | chemistry_check_logger
    | JsonOutputParser()
)

# Prompt for chemistry question generation
generation_prompt = generation_prompts.generation_prompt

# LLM nature query detection
self_nature_check_llm = ChatOllama(
    base_url=url,
    model=mdl,
    format="json",
    temperature=0,
    client_kwargs={"headers": auth_headers},
)

# Prompt for LLM nature question checking
llm_nature_checking_prompt = query_handling_prompts.llm_nature_checking_prompt
llm_nature_check_logger = make_log_usage_node("LLM Nature Checker")

# Chain for LLM nature query checking
llm_nature_query_checker = (
    llm_nature_checking_prompt
    | self_nature_check_llm
    | llm_nature_check_logger
    | JsonOutputParser()
)

# Greeting detection
greeting_check_llm = ChatOllama(
    base_url=url,
    model=mdl,
    format="json",
    temperature=0,
    client_kwargs={"headers": auth_headers},
)

# Prompt for greeting checking
greeting_checking_prompt = query_handling_prompts.greeting_checking_prompt
greeting_check_logger = make_log_usage_node("Greeting Checker")

# Chain for greeting detection
greeting_query_checker = (
    greeting_checking_prompt
    | greeting_check_llm
    | greeting_check_logger
    | JsonOutputParser()
)

# Summarization detection
summarization_check_llm = ChatOllama(
    base_url=url,
    model=mdl,
    format="json",
    temperature=0,
    num_ctx=8000,
    client_kwargs={"headers": auth_headers},
)

# Prompt for summarization checking
summarization_checking_prompt = query_handling_prompts.summarization_checking_prompt
summarization_check_logger = make_log_usage_node("Summarization Checker")

# Chain for summarization detection
summarization_query_checker = (
    summarization_checking_prompt
    | summarization_check_llm
    | summarization_check_logger
    | JsonOutputParser()
)

# List document query detection
list_documents_query_check_llm = ChatOllama(
    base_url=url,
    model=mdl,
    format="json",
    temperature=0,
    num_ctx=8000,
    client_kwargs={"headers": auth_headers},
)

# Prompt for list document query checking
list_documents_query_checking_prompt = (
    query_handling_prompts.list_documents_query_checking_prompt
)
list_documents_query_check_logger = make_log_usage_node("Listing of Documents Checker")

# Chain for list document query detection
list_documents_query_checker = (
    list_documents_query_checking_prompt
    | list_documents_query_check_llm
    | list_documents_query_check_logger
    | JsonOutputParser()
)

# Obtain patent classification LLM
obtain_patent_classification_llm = ChatOllama(
    base_url=url,
    model=mdl,
    format="json",
    temperature=0,
    num_ctx=8000,
    client_kwargs={"headers": auth_headers},
)

# Check user query again with LLM to identify if a classification-based query can be made
obtain_patent_classification_prompt = (
    topic_classification_prompts.obtain_patent_classification_prompt
)
obtain_patent_classification_logger = make_log_usage_node(
    "Obtain Patent Classification"
)

# Chain for obtaining patent classification
obtain_patent_classification_chain = (
    obtain_patent_classification_prompt
    | obtain_patent_classification_llm
    | obtain_patent_classification_logger
    | JsonOutputParser()
)

# Generic output generation
gen_int_out_llm = ChatOllama(
    base_url=url,
    temperature=0.0,
    model=gpt_oss_mdl,
    client_kwargs={"headers": auth_headers},
    tags=["gen_int"],
)


# Dynamic prompt template selection based on model name
def get_gen_int_out_prompt(model_name: str):
    """Select appropriate prompt template based on model name"""
    if "gpt" in model_name.lower():
        return generation_prompts.gen_int_out_prompt_gpt
    else:
        return generation_prompts.gen_int_out_prompt_llama


# Get the appropriate prompt template for the current model
gen_int_out_prompt = get_gen_int_out_prompt(gen_int_out_llm.model)
gen_int_out_logger = make_log_usage_node("Generation with Inherent LLM Knowledge")

# Chain for generic output
gen_int_out_chain = (
    gen_int_out_prompt | gen_int_out_llm | gen_int_out_logger | StrOutputParser()
)


# Graph state definition
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    What this class does:
        Defines the structure and types for the state object that gets passed
        between different nodes in the graph workflow.

    Attributes:
        question: The current query being processed.
        generation: Generated answer content from LLM or RAG.
        documents: List of retrieved documents before filtering.
        filtered_docs: List of documents after relevance filtering.
        user_decision: Any decision or feedback from the user.
        corpus_type: Type of corpus being queried (patent, literature, tds, sds).
        data_source: Specific data source or classification within the corpus.
        document_sources: List of source identifiers for the documents.
        messages: Chat history with annotations for adding new messages.
        document_filter: Optional document type filter.
        timings: Dictionary containing timing information for various operations.
    """

    question: str
    generation: str
    documents: List[str]
    filtered_docs: List[str]
    user_decision: str
    corpus_type: str
    data_source: str
    document_sources: List[str]
    messages: Annotated[Sequence[BaseMessage], add_messages]
    document_filter: Optional[str]
    timings: dict
    listing_context: Optional[dict] 


@timed_step("reformulate_question", tag="query_handling")
def reformulate_question(state, writer: StreamWriter):
    """
    Use the existing chat history to reformulate the user query.

    What this function does:
        Reformulates ambiguous or context-dependent user queries into standalone
        queries based on conversation history.

    Variables:
        - REFORMULATE_QUESTION_FLAG: Configuration flag to enable/disable reformulation.
        - messages: The full chat history from the graph state.
        - question: The latest user query extracted from messages.
        - chat_history: Previous messages (up to 5) used for context.
        - reformulated_question: The query after reformulation.
        - standalone_detection: Result of checking if query is already standalone.

    Logic:
        1. Extracts the latest user query from the message history
        2. Gets relevant chat history (up to 5 most recent messages)
        3. If reformulation is enabled and chat history exists:
           - Logs the original question
           - Checks if the query is already a standalone query
           - Always reformulates the query for consistency (temporary solution)
        4. If no chat history exists or reformulation is disabled:
           - Uses the original query without changes
        5. Logs the reformulated question for debugging
        6. Returns the state with the updated question

    Returns:
        dict: Updated state with the reformulated question.
    """
    writer("Understanding your question ...")
    try:
        print("--" * 10 + "\n")
        REFORMULATE_QUESTION_FLAG = Config.REFORMULATE_QUESTION_FLAG
        NUMBER_OF_PREVIOUS_MESSAGES_TO_CONSIDER = 20
        messages = state["messages"]
        question = messages[-1].content
        chat_history = []

        def get_message_role(message: BaseMessage) -> str:
            """Get the role of a message based on its type"""
            if isinstance(message, HumanMessage):
                return "Human"
            if isinstance(message, SystemMessage):
                return "System"
            elif isinstance(message, AIMessage):
                return "Assistant"
            else:
                # For other message types (SystemMessage, etc.)
                return message.__class__.__name__.replace("Message", "")

        def get_formatted_chat_history(
            messages: list[BaseMessage], num_messages: int = 4
        ) -> str:
            """Format chat history - just take the last N messages with roles"""
            if len(messages) < 2:
                return ""

            # Remove current question and take last N previous messages
            previous_messages = messages[-(num_messages + 1) : -1]

            formatted_messages = []
            for msg in previous_messages:
                role = get_message_role(msg)
                if not role == "System":
                    formatted_messages.append(f"{role}: {msg.content}")
            return "\n\n".join(formatted_messages)

        if messages:
            # Use only the last N messages for reformulation. If there are fewer than N messages, use all messages.
            chat_history = get_formatted_chat_history(
                messages, NUMBER_OF_PREVIOUS_MESSAGES_TO_CONSIDER
            )
            # print(f"Chat History:\n{chat_history}")

        reformulated_question = None

        if not chat_history:
            print(f"No chat history found. Using original query: {question}")
            return {"question": question}

        if REFORMULATE_QUESTION_FLAG:
            print(f"\nUser Query: {question}")
            print(f"Chat History Length: {len(messages)} messages")

            # Step 2: Check if standalone
            standalone_detection = standalone_detection_chain.invoke(
                {"input": question, "chat_history": chat_history}
            )

            print("Standalone Detection : ", "\n\n", standalone_detection, "\n\n")

            if standalone_detection["standalone_query"]:
                # Related topic but already standalone
                reformulated_question = question
                print(
                    f"Question is already standalone. Using original query: {reformulated_question}"
                )
            else:
                # Related topic and needs reformulation
                reformulated_question = reformulation_chain.invoke(
                    {"input": question, "chat_history": chat_history}
                )
                print(
                    f"Question has been reformulated for context.\nReformulated query: {reformulated_question}"
                )
        else:
            # If reformulation is disabled, just use the original question
            print(f"Query reformulation is disabled. Using original query: {question}")
            reformulated_question = question

        # Return the new state with the reformulated question
        return {"question": reformulated_question}
    except Exception as e:
        print(f"Error in reformulation: {e}")
        # Fallback to original question
        messages = state["messages"]
        return {"question": messages[-1].content}


def decide_patent_database(state, writer: StreamWriter):
    """
    Routes to proper patent database for retrieval.

    What this function does:
        Analyzes the user query and determines which patent database classification
        is most appropriate for finding relevant information.

    Variables:
        - question: The user query to analyze.
        - data_source: The determined database classification.

    Logic:
        1. Logs the start of the decision process
        2. Extracts the user query from the state
        3. Invokes the patent_question_router to classify the query into one of:
           - "Polymers", "Glass", "Composites", or "Unknown"
        4. Logs the selected database classification
        5. Returns the updated state with data_source and corpus_type

    Returns:
        dict: Updated state with data_source set to the appropriate database classification
              and corpus_type set to "patent".
    """
    writer("Scanning patent database ...")
    print("--" * 10, "Deciding my knowledge source", "--" * 10)
    question = state["question"]
    data_source = patent_question_router.invoke(question)
    print(
        f"{'--' * 10} Retrieving from {data_source['datasource']} database {'--' * 10}"
    )
    return {"data_source": data_source["datasource"], "corpus_type": "patent"}


def route_to_patent_database(state, writer: StreamWriter):
    """Retrieve documents from relevant patent data source.

    What this function does:
        Fetches relevant documents from the appropriate patent database based on
        the data source determined in the previous step.

    Variables:
        - source: The specific patent database classification to query.
        - question: The user query to search with.
        - corpus_key: The type of corpus being queried (set to "patents").
        - retriever: The retriever instance configured for this query.
        - documents: The retrieved document chunks.

    Logic:
        1. Extracts the data source and query from the state
        2. Logs the start of document retrieval
        3. Checks if the source is valid for the patents collection:
           - If valid, attempts to retrieve documents:
             * Creates an appropriate retriever with reranking if configured
             * Invokes the retriever with the query
             * Logs the number of retrieved documents
             * Returns the updated state with documents
           - If retrieval fails, returns an error message
           - If the source is invalid, returns a "learning this domain" message

    Returns:
        dict: Updated state with either:
              - The retrieved documents added to the "documents" field, or
              - An error message in the "generation" field and empty documents list
    """
    writer(" Using patent database as knowledge source ...")
    source = state["data_source"]
    question = state["question"]
    corpus_key = "patents"

    print(f"{'--' * 10} Obtaining documents relevant to question {'--' * 10}")

    if source in COLLECTIONS_CONFIG()[corpus_key]["classifications"]:
        try:
            # Get appropriate retriever based on configuration
            retriever = get_retriever(
                state,
                collection_type=corpus_key,
                classification=source,
                query=question,
                use_reranking=Config.USE_RERANKING,
            )

            # Invoke retriever with appropriate parameters
            start = time.perf_counter()
            documents = retriever.invoke(question)
            end = time.perf_counter()
            duration = end - start
            timings = state.get("timings")
            timings["retriever"] = {
                "duration_sec": round(duration, 6),
                "start_time": start,
                "end_time": end,
                "tag": "reranked retrieval",
            }
            state["timings"] = timings

            print(
                f"---RETRIEVED {len(documents)} DOCUMENT CHUNKS FROM PATENTS COLLECTION---"
            )
            return {"documents": documents}

        except Exception as e:
            print(f"Error during PATENT retrieval: {e.__class__.__name__}")
            print(f"Error message: {str(e)}")
            return {
                "documents": [],
                "generation": "I apologize, but I encountered an error while obtaining documents relevant to your query. Please try again or rephrase your question.",
            }
    else:
        print("I am learning this domain.")
        return {"documents": [], "generation": "I am learning this domain."}


@timed_step("obtain_list_of_filtered_patents", tag="answer_generation")
def obtain_list_of_filtered_patents_legacy(state, writer: StreamWriter):
    """
    LEGACY IMPLEMENTATION: Non-agentic document listing.
    
    Returns first 15 patents sorted by publication date.
    Used when USE_AGENTIC_LISTING = False.
    """
    # TODO: Break this function into smaller reusable functions
    question = state["question"]
    corpus_key = "patents"

    print(f"{'--' * 10} Obtaining list of patents relevant to user query {'--' * 10}")
    NUMBER_OF_PATENTS_IDS_TO_BE_PRINTED = 15

    # Get structured filter if using self query
    _, translated_filter = create_milvus_filter(state, corpus_key, question)
    print(f"- Translated filter: {translated_filter}")

    obtain_patent_classification_separately = (
        Config.OBTAIN_PATENT_CLASSIFICATION_SEPARATELY
    )

    if obtain_patent_classification_separately and (
        translated_filter is None
        or "classifications like" not in translated_filter["expr"].lower()
    ):
        print(
            "No classifications based filtering found in the original milvus filter. Trying again to obtain a classification based filter..."
        )
        possible_classifications = obtain_patent_classification_chain.invoke(question)

        def create_like_clause(
            classifications: list[str], column_name: str = "classifications"
        ) -> str:
            # Create LIKE conditions for each classification
            like_conditions = [
                f'{column_name} LIKE "%{classification}%"'
                for classification in classifications
            ]

            # Join with OR and wrap in parentheses
            return f"({' OR '.join(like_conditions)})"

        if possible_classifications["classifications"] != ["None"]:
            classification_milvus_filter_string = create_like_clause(
                possible_classifications["classifications"], "classifications"
            )
            print(
                f"Constructed classification filter: {classification_milvus_filter_string}"
            )

            # Insert the classification filter into the translated filter
            # Translated filter is a dictionary with "expr" key and looks like {'expr': '(( inventor_name like "%Michael Johnson%" ) and ( expiration_year <= 2038 ))'} or {'expr': '( expiration_year < 2039 )'}
            if translated_filter is None:
                translated_filter = {"expr": classification_milvus_filter_string}
            else:
                modified_expr = translated_filter["expr"]
                # If the expr already has exactly one starting parentheses, we can just append the classification filter and add parentheses around the whole expression
                if modified_expr.startswith("(") and modified_expr.count("(") == 1:
                    modified_expr = (
                        f"({modified_expr} and {classification_milvus_filter_string})"
                    )
                elif modified_expr.endswith("))") and modified_expr.count(")") > 1:
                    modified_expr = (
                        # f"{modified_expr[:-1]} and {classification_milvus_filter_string})"
                        f"({modified_expr} and {classification_milvus_filter_string})"
                    )
                translated_filter["expr"] = modified_expr

            use_entity_matching_flag = Config.USE_ENTITY_MATCHING_FOR_PATENTS
            attributes_to_expand = [
                "classifications",
            ]

            # For patent collection, expand categories with semantically similar ones
            if (
                translated_filter
                and "expr" in translated_filter
                and use_entity_matching_flag
            ):
                expanded_filter = expand_milvus_filter_through_attribute_matching(
                    translated_filter,
                    question,
                    attributes_to_expand,
                    Config.get_dynamic_collection_name(corpus_key),
                )
                translated_filter = expanded_filter
        else:
            print(
                "No classifications obtained via analyzing the user query again. Using original filter."
            )

    if translated_filter is not None and "expr" in translated_filter:
        # Get patent data (patent ID, title, publication date)
        patent_data_tuples = get_unique_attribute_values(
            collection_name=Config.get_dynamic_collection_name(corpus_key),
            attribute_name=["patent", "title", "pub_date", "url"],
            query=translated_filter["expr"],
        )

        number_of_unique_patents = len(patent_data_tuples)
        print(
            f"Number of unique patents that satisfy user query: {number_of_unique_patents}"
        )

        if number_of_unique_patents == 0:
            generation = "No patents found that satisfy your query. Please modify your question or try to phrase it in an alternative manner."
            return {
                "generation": generation,
                "messages": [AIMessage(content=generation)],
            }
        print("\n\n : Patent Data Tuples : \n",patent_data_tuples,"\n\n")
        # Sort by publication date (most recent first)
        # Handle potential date parsing issues
        def parse_date_for_sorting(date_str):
            """Parse date string for sorting, handling various formats"""
            if not date_str or date_str is None:
                return "1900-01-01"  # Default for None/empty dates

            try:
                # Common formats to try
                date_formats = [
                    "%Y-%m-%d",  # 2023-12-31
                ]

                for fmt in date_formats:
                    try:
                        parsed_date = datetime.strptime(str(date_str).strip(), fmt)
                        return parsed_date.strftime("%Y-%m-%d")
                    except ValueError:
                        continue

                # If no format works, return the original string
                return str(date_str)

            except Exception:
                return str(date_str) if date_str else "1900-01-01"

        # Sort patents by publication date (most recent first)
        try:
            sorted_patent_data = sorted(
                patent_data_tuples,
                key=lambda x: (
                    parse_date_for_sorting(x[2]) if len(x) > 2 else "1900-01-01"
                ),
                reverse=True,  # Most recent first
            )
        except Exception as e:
            print(f"Warning: Could not sort by date, using original order. Error: {e}")
            sorted_patent_data = patent_data_tuples

        # Print sample data for debugging
        print(
            f"First {min(NUMBER_OF_PATENTS_IDS_TO_BE_PRINTED, len(sorted_patent_data))} patents (sorted by publication date):"
        )
        for i, (patent_id, title, pub_date, patent_url) in enumerate(
            sorted_patent_data[:NUMBER_OF_PATENTS_IDS_TO_BE_PRINTED]
        ):
            print(f"  {i + 1}. {patent_id} | {title[:100]}... | {pub_date}")

        print("---LISTING PATENTS THAT SATISFY USER QUERY. PRINTING...---")
        number_of_unique_patents_that_can_be_printed = min(
            NUMBER_OF_PATENTS_IDS_TO_BE_PRINTED, number_of_unique_patents
        )

        if number_of_unique_patents_that_can_be_printed == number_of_unique_patents:
            pretext = f"There are {number_of_unique_patents} patents that satisfy your criteria:\n\n"
            posttext = (
                "\n\nFeel free to ask for more details about any specific patent(s)."
            )
        else:
            pretext = f"There are {number_of_unique_patents} unique patents that satisfy your criteria. Here is the list of {number_of_unique_patents_that_can_be_printed} most recently published patents:\n\n"
            posttext = f"\n\nNote: This list shows the {number_of_unique_patents_that_can_be_printed} most recently published patents from {number_of_unique_patents} total patents that satisfy your query. Please narrow down your question to obtain more detailed information about any specific patent(s)."

        list_of_patents = ""
        for i, (patent_id, title, pub_date, patent_url) in enumerate(
            sorted_patent_data[:number_of_unique_patents_that_can_be_printed]
        ):
            # Clean up title if it's too long
            display_title = title[:500] + "..." if title and len(title) > 500 else title
            list_of_patents += (
                f"{i + 1}. [{patent_id}]({patent_url}) - {display_title}\n"
            )

        generation = pretext + list_of_patents + posttext

        return {
            "generation": generation,
            "messages": [AIMessage(content=generation)],
        }
    else:
        generation = "I am unable to answer your question. Please narrow down your question or try to phrase it in an alternative manner."
        return {
            "generation": generation,
            "messages": [AIMessage(content=generation)],
        }











# UTILITY FUNCTIONS

def get_document_type_name(corpus_type: str, plural: bool = True) -> str:
    """
    Get human-readable document type name from corpus type.
    
    Args:
        corpus_type: Corpus type key ("patents", "literature", "tds")
        plural: If True, return plural form; if False, return singular
        
    Returns:
        Human-readable document type name
    """
    if plural:
        return {
            "patents": "patents",
            "literature": "papers",
            "tds": "technical datasheets"
        }.get(corpus_type, "documents")
    else:
        return {
            "patents": "patent",
            "literature": "paper", 
            "tds": "technical datasheet"
        }.get(corpus_type, "document")


# COMPONENT INITIALIZATION

def initialize_listing_components(corpus_key: str) -> Tuple[ChatOllama, ChatOllama, ListingCoordinator]:
    """
    Initialize LLM models and coordinator for listing.
    
    Args:
        corpus_key: Corpus type ("patents", "literature", "tds")
        
    Returns:
        Tuple of (analyzer_llm, formatter_llm, coordinator)
    """
    url = st.secrets["model_endpoints"]["matquest_endpoint"]
    key = st.secrets["model_authentication"]["encoded_key"]
    auth_headers = {"Authorization": f"Basic {key}"}
    
    analyzer_llm = ChatOllama(
        base_url=url,
        model=Config.LISTING_AGENT_MODEL,
        reasoning=True,
        format="json",
        temperature=0,
        num_ctx=10000,
        client_kwargs={"headers": auth_headers},
    )
    
    formatter_llm = ChatOllama(
        base_url=url,
        model=Config.LISTING_FORMATTER_MODEL,
        temperature=0.0,
        num_ctx=22000,
        reasoning=True,
        tags=["listing"],
        client_kwargs={"headers": auth_headers},
    )
    
    coordinator = ListingCoordinator(corpus_key, analyzer_llm)
    
    return analyzer_llm, formatter_llm, coordinator


# CORPUS-SPECIFIC FILTER GENERATION & DOCUMENT FETCHING

def generate_filter_and_fetch_documents_patents(
    state: GraphState,
    corpus_key: str,
    question: str
) -> Tuple[list, Dict]:
    """
    Generate filter and fetch documents for PATENTS (corpus-specific).
    
    This is designed to run in parallel with query analysis.
    
    Args:
        state: GraphState
        corpus_key: "patents"
        question: User query
        
    Returns:
        Tuple of (all_documents, translated_filter)
    """
    print("📋 Filter Generation & Document Retrieval (Patents):")
    
    # Step 1: Create initial Milvus filter
    _, translated_filter = create_milvus_filter(state, corpus_key, question)
    print(f"🔍 Initial filter: {translated_filter}")
    
    # Step 2: Handle patent classifications separately (PATENT-SPECIFIC)
    if Config.OBTAIN_PATENT_CLASSIFICATION_SEPARATELY and (
        translated_filter is None or 
        "classifications like" not in translated_filter["expr"].lower()
    ):
        print("🔍 Extracting classifications separately...")
        possible_classifications = obtain_patent_classification_chain.invoke(question)
        
        def create_like_clause(
            classifications: list[str], column_name: str = "classifications"
        ) -> str:
            like_conditions = [
                f'{column_name} LIKE "%{classification}%"'
                for classification in classifications
            ]
            return f"({' OR '.join(like_conditions)})"
        
        if possible_classifications["classifications"] != ["None"]:
            classification_milvus_filter_string = create_like_clause(
                possible_classifications["classifications"], "classifications"
            )
            print(f"   Constructed: {classification_milvus_filter_string}")
            
            # Merge with existing filter
            if translated_filter is None:
                translated_filter = {"expr": classification_milvus_filter_string}
            else:
                modified_expr = translated_filter["expr"]
                if modified_expr.startswith("(") and modified_expr.count("(") == 1:
                    modified_expr = f"({modified_expr} and {classification_milvus_filter_string})"
                elif modified_expr.endswith("))") and modified_expr.count(")") > 1:
                    modified_expr = f"({modified_expr} and {classification_milvus_filter_string})"
                translated_filter["expr"] = modified_expr
            
            # Entity matching expansion (PATENT-SPECIFIC)
            if Config.USE_ENTITY_MATCHING_FOR_PATENTS:
                expanded_filter = expand_milvus_filter_through_attribute_matching(
                    translated_filter,
                    question,
                    ["classifications"],
                    Config.get_dynamic_collection_name(corpus_key)
                )
                translated_filter = expanded_filter
                print(f"   Expanded filter: {translated_filter['expr'][:100]}...")
    
    # Validate filter
    if not translated_filter or "expr" not in translated_filter:
        raise ValueError("Failed to generate valid filter for patents")
    
    print(f"✅ Final filter: {translated_filter['expr'][:100]}...")
    
    # Step 3: Fetch ALL documents with ALL attributes
    print("📥 Fetching documents from Milvus...")
    all_documents = get_unique_attribute_values(
        collection_name=Config.get_dynamic_collection_name(corpus_key),
        attribute_name=DocumentListingPrompts.CORPUS_ATTRIBUTES[corpus_key]["all_available"],
        query=translated_filter["expr"]
    )
    
    print(f"✅ Retrieved {len(all_documents)} documents")
    
    return all_documents, translated_filter


def generate_filter_and_fetch_documents_literature(
    state: GraphState,
    corpus_key: str,
    question: str
) -> Tuple[list, Dict]:
    """
    Generate filter and fetch documents for LITERATURE (corpus-specific).
    
    Args:
        state: GraphState
        corpus_key: "literature"
        question: User query
        
    Returns:
        Tuple of (all_documents, translated_filter)
    """
    print("📋 Filter Generation & Document Retrieval (Literature):")
    
    # Literature-specific filter generation (simpler than patents)
    _, translated_filter = create_milvus_filter(state, corpus_key, question)
    print(f"🔍 Initial filter: {translated_filter}")
    
    # Validate filter
    if not translated_filter or "expr" not in translated_filter:
        raise ValueError("Failed to generate valid filter for literature")
    
    print(f"✅ Final filter: {translated_filter['expr'][:100]}...")
    
    # Fetch documents
    print("📥 Fetching documents from Milvus...")
    all_documents = get_unique_attribute_values(
        collection_name=Config.get_dynamic_collection_name(corpus_key),
        attribute_name=DocumentListingPrompts.CORPUS_ATTRIBUTES[corpus_key]["all_available"],
        query=translated_filter["expr"]
    )
    
    print(f"✅ Retrieved {len(all_documents)} documents")
    
    return all_documents, translated_filter


def generate_filter_and_fetch_documents_tds(
    state: GraphState,
    corpus_key: str,
    question: str
) -> Tuple[list, Dict]:
    """
    Generate filter and fetch documents for TDS (corpus-specific).
    
    Args:
        state: GraphState
        corpus_key: "tds"
        question: User query
        
    Returns:
        Tuple of (all_documents, translated_filter)
    """
    print("📋 Filter Generation & Document Retrieval (TDS):")
    
    # TDS-specific filter generation
    _, translated_filter = create_milvus_filter(state, corpus_key, question)
    print(f"🔍 Initial filter: {translated_filter}")
    
    # Validate filter
    if not translated_filter or "expr" not in translated_filter:
        raise ValueError("Failed to generate valid filter for TDS")
    
    print(f"✅ Final filter: {translated_filter['expr'][:100]}...")
    
    # Fetch documents
    print("📥 Fetching documents from Milvus...")
    all_documents = get_unique_attribute_values(
        collection_name=Config.get_dynamic_collection_name(corpus_key),
        attribute_name=DocumentListingPrompts.CORPUS_ATTRIBUTES[corpus_key]["all_available"],
        query=translated_filter["expr"]
    )
    
    print(f"✅ Retrieved {len(all_documents)} documents")
    
    return all_documents, translated_filter


# EDGE CASE HANDLING

def handle_edge_cases(
    documents_to_display: list,
    is_followup: bool,
    cumulative_shown: int,
    total_available: int,
    corpus_key: str,
    listing_context: Optional[Dict]
) -> Optional[Dict]:
    """
    Handle edge cases for empty results.
    
    Args:
        documents_to_display: List of documents to display
        is_followup: Whether this is a follow-up query
        cumulative_shown: Total documents shown so far
        total_available: Total documents available
        corpus_key: Corpus type
        listing_context: Current listing context
        
    Returns:
        Dictionary with generation and messages if edge case handled, None otherwise
    """
    doc_type_plural = get_document_type_name(corpus_key, plural=True)
    
    # Case 1: All documents already shown (follow-up with nothing more)
    if not documents_to_display and is_followup and cumulative_shown >= total_available > 0:
        print(f"\n{'='*70}")
        print(f"📌 EDGE CASE: All {total_available} documents already shown")
        print(f"{'='*70}")
        
        # Keep context for transparency
        if listing_context:
            st.session_state.listing_context = listing_context
        
        final_output = (
            f"✓ **All {total_available} matching {doc_type_plural} have already been displayed.**\n\n"
            f"You've seen the complete list of results. "
            f"Start a new search to explore different {doc_type_plural}."
        )
        
        print("✅ Completion message prepared")
        print(f"{'='*70}")
        print(f"🤖 AGENTIC {corpus_key.upper()} LISTING - COMPLETE")
        print(f"{'='*70}\n")
        
        return {
            "generation": final_output,
            "messages": [AIMessage(content=final_output)],
        }
    
    # Case 2: Initial query with no results
    if not documents_to_display and not is_followup and total_available == 0:
        print(f"\n{'='*70}")
        print(f"📌 EDGE CASE: No documents found")
        print(f"{'='*70}")
        
        # Clear context
        st.session_state.listing_context = None
        
        final_output = (
            f"No {doc_type_plural} were found matching your search criteria.\n\n"
            f"**Suggestions:**\n"
            f"- Try broadening your search terms\n"
            f"- Use different keywords or synonyms\n"
            f"- Check spelling of technical terms"
        )
        
        print("✅ No results message prepared")
        print(f"{'='*70}")
        print(f"🤖 AGENTIC {corpus_key.upper()} LISTING - COMPLETE")
        print(f"{'='*70}\n")
        
        return {
            "generation": final_output,
            "messages": [AIMessage(content=final_output)],
        }
    
    # Case 3: Unexpected empty result (defensive)
    if not documents_to_display and total_available > 0:
        print(f"\n{'='*70}")
        print(f"⚠️  WARNING: Unexpected empty result")
        print(f"{'='*70}")
        
        final_output = (
            f"An error occurred while retrieving results. "
            f"Please try your request again or start a new search."
        )
        
        return {
            "generation": final_output,
            "messages": [AIMessage(content=final_output)],
        }
    
    return None


# CONTEXT MANAGEMENT

def manage_listing_context(
    tool_result: Dict,
    translated_filter: Dict,
    parsed_query: Dict,
    listing_context: Optional[Dict],
    corpus_key: str
) -> Dict:
    """
    Create or update listing context.
    
    Args:
        tool_result: Result from tool execution
        translated_filter: Milvus filter used
        parsed_query: Parsed query requirements
        listing_context: Existing context (if any)
        corpus_key: Corpus type
        
    Returns:
        Updated or new listing context
    """
    print(f"\n{'-'*70}")
    print("💾 CONTEXT MANAGEMENT")
    print(f"{'-'*70}")
    
    if tool_result.get('tool_used') == "continue":
        # Update existing context
        listing_context = ListingContextManager.update_context(
            listing_context, 
            tool_result,
            translated_filter["expr"],
            parsed_query
        )
        print("🔄 Context updated (continuation)")
    else:
        # Create new context
        listing_context = ListingContextManager.create_context(
            translated_filter["expr"], 
            tool_result, 
            parsed_query
        )
        print("🆕 New context created")
    
    st.session_state.listing_context = listing_context
    
    print(f"📦 Context details:")
    print(f"   • Filter: {listing_context['filter_expr'][:50]}...")
    print(f"   • Shown: {listing_context['shown_count']}/{listing_context['total_count']}")
    print(f"   • Timestamp: {listing_context['timestamp']}")
    
    return listing_context


# OUTPUT FORMATTING

def format_listing_output(
    tool_result: Dict,
    parsed_query: Dict,
    formatter_llm: ChatOllama,
    corpus_key: str,
    writer: StreamWriter,
    original_query: str
) -> str:
    """
    Format tool output into readable markdown.
    
    Args:
        tool_result: Result from tool execution
        parsed_query: Parsed query requirements
        formatter_llm: LLM for formatting
        corpus_key: Corpus type
        writer: StreamWriter for progress
        
    Returns:
        Formatted output string
    """
    writer("Formatting your results...")
    print(f"\n{'-'*70}")
    print("✨ OUTPUT FORMATTING")
    print(f"{'-'*70}")
    
    formatter = ListingOutputFormatter(formatter_llm, corpus_key)
    formatted_output = formatter.format_output(
        documents=tool_result['documents_to_display'],
        tool_result=tool_result,
        parsed_query=parsed_query,
        is_followup=(tool_result.get('tool_used') == "continue"),
        original_query=original_query,
        
    )
    
    print("✅ Output formatted successfully")
    
    return formatted_output


# MAIN LISTING WORKFLOW EXECUTION

def execute_listing_workflow(
    state: GraphState,
    writer: StreamWriter,
    corpus_key: str,
    filter_fetch_function: Callable
) -> Dict:
    """
    Common listing workflow for all corpus types.
    
    This orchestrates the complete listing process:
    1. Initialize components
    2. Parallel execution (filter generation + query analysis)
    3. Validation
    4. Tool execution
    5. Edge case handling
    6. Context management
    7. Output formatting
    
    Args:
        state: GraphState containing question and other data
        writer: StreamWriter for progress messages
        corpus_key: Corpus type ("patents", "literature", "tds")
        filter_fetch_function: Corpus-specific filter generation function
        
    Returns:
        Dictionary with generation and messages
    """
    question = state["question"]
    listing_context = st.session_state.get("listing_context")
    doc_type_plural = get_document_type_name(corpus_key, plural=True)
    
    print(f"\n{'='*70}")
    print(f"🤖 AGENTIC {corpus_key.upper()} LISTING - START")
    print(f"{'='*70}")
    print(f"📝 Query: {question}")
    print(f"🗂️  Existing context: {'Yes' if listing_context else 'No'}")
    
    try:
        # STEP 1: Initialize Components
        writer(f"Analyzing your {doc_type_plural} request...")
        print("\n✅ Initializing components...")
        
        analyzer_llm, formatter_llm, coordinator = initialize_listing_components(corpus_key)
        
        print("✅ Components initialized")
        
        # STEP 2: Parallel Execution
        writer("Filtering and analyzing...")
        print(f"\n{'-'*70}")
        print("⚡ PARALLEL EXECUTION")
        print(f"{'-'*70}")
        
        start_parallel = time.perf_counter()
        
        if Config.USE_PARALLEL_EXECUTION:
            print("🚀 Running tasks in parallel:")
            print("   • Task A: Filter generation + Document retrieval")
            print("   • Task B: Query analysis")
            
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Task A: Complete filter generation + fetch documents
                future_docs_filter = executor.submit(
                    filter_fetch_function,
                    state,
                    corpus_key,
                    question
                )
                
                # Task B: Analyze query intent
                future_parsed_query = executor.submit(
                    coordinator.query_analyzer.analyze_query,
                    question,
                    listing_context
                )
                
                # Wait for both to complete
                all_documents, translated_filter = future_docs_filter.result()
                parsed_query = future_parsed_query.result()
        else:
            # Sequential execution (fallback)
            print("🔄 Running tasks sequentially (parallel execution disabled)")
            all_documents, translated_filter = filter_fetch_function(
                state, corpus_key, question
            )
            parsed_query = coordinator.query_analyzer.analyze_query(
                question, listing_context
            )
        
        end_parallel = time.perf_counter()
        print(f"✅ Parallel execution complete in {end_parallel - start_parallel:.2f}s")
        print(f"   • Retrieved: {len(all_documents)} documents")
        print(f"   • Parsed query: {parsed_query}")
        
        # STEP 3: Validate Results
        print(f"\n{'-'*70}")
        print("🔍 VALIDATION")
        print(f"{'-'*70}")
        
        # Check for empty results
        if len(all_documents) == 0:
            empty_msg = f"No {doc_type_plural} found matching your criteria. Please try broadening your search."
            print("❌ No documents found")
            st.session_state.listing_context = None
            return {
                "generation": empty_msg,
                "messages": [AIMessage(content=empty_msg)],
            }
        
        # Validate existing context
        if listing_context:
            should_reset = coordinator.should_reset_context(
                parsed_query, 
                listing_context,
                translated_filter["expr"]
            )
            
            if should_reset:
                print("🔄 Context reset triggered")
                print(f"   Reason: Filter/sort change or timeout")
                listing_context = None
            else:
                print("✅ Context valid - continuing pagination")
                print(f"   • Previously shown: {listing_context.get('shown_count', 0)}")
                print(f"   • Total available: {listing_context.get('total_count', 0)}")
        else:
            print("🆕 No existing context - new listing")
        
        # STEP 4: Tool Execution
        writer("Organizing results...")
        print(f"\n{'-'*70}")
        print("🔧 TOOL SELECTION & EXECUTION")
        print(f"{'-'*70}")
        
        tool_result = coordinator.execute_listing(
            all_documents=all_documents,
            parsed_query=parsed_query,
            listing_context=listing_context
        )
        
        # Extract key results
        documents_to_display = tool_result.get("documents_to_display", [])
        has_more = tool_result.get("has_more", False)
        cumulative_shown = tool_result.get("cumulative_shown", 0)
        total_available = tool_result.get("total_available", 0)
        is_followup = parsed_query.get("is_followup", False)
        
        print(f"✅ Tool executed: {tool_result.get('tool_used')}")
        print(f"📊 Results:")
        print(f"   • Shown: {tool_result['shown_count']}")
        print(f"   • Total: {tool_result['total_available']}")
        print(f"   • Attributes: {tool_result.get('attributes_included', [])}")
        print(f"   • Sort: {tool_result.get('sort_applied')} {tool_result.get('sort_order')}")
        
        # STEP 5: Handle Edge Cases
        edge_case_result = handle_edge_cases(
            documents_to_display,
            is_followup,
            cumulative_shown,
            total_available,
            corpus_key,
            listing_context
        )
        
        if edge_case_result:
            return edge_case_result
        
        # STEP 6: Context Management
        listing_context = manage_listing_context(
            tool_result,
            translated_filter,
            parsed_query,
            listing_context,
            corpus_key
        )
        
        # STEP 7: Format Output
        formatted_output = format_listing_output(
            tool_result,
            parsed_query,
            formatter_llm,
            corpus_key,
            writer,
            question
        )
        
        print(f"{'='*70}")
        print(f"🤖 AGENTIC {corpus_key.upper()} LISTING - COMPLETE")
        print(f"{'='*70}\n")
        
        return {
            "generation": formatted_output,
            "messages": [AIMessage(content=formatted_output)],
        }
    
    except Exception as e:
        # ====================================================================
        # Error Handling
        # ====================================================================
        print(f"\n{'='*70}")
        print(f"❌ ERROR IN AGENTIC {corpus_key.upper()} LISTING")
        print(f"{'='*70}")
        print(f"Error type: {e.__class__.__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"{'='*70}\n")
        
        # Determine appropriate error message
        error_lower = str(e).lower()
        
        if "filter" in error_lower or "query" in error_lower:
            error_msg = "I couldn't understand the filtering criteria in your query. Please try rephrasing."
        elif "milvus" in error_lower or "database" in error_lower or "connection" in error_lower:
            error_msg = "I encountered a database error. Please try again in a moment."
        elif "timeout" in error_lower:
            error_msg = "The request took too long. Please try narrowing your search criteria."
        elif "llm" in error_lower or "model" in error_lower:
            error_msg = "I encountered an issue with the language model. Please try again."
        else:
            error_msg = f"I encountered an error processing your {doc_type_plural} listing request. Please try rephrasing or simplifying your query."
        
        return {
            "generation": error_msg,
            "messages": [AIMessage(content=error_msg)],
        }


@timed_step("obtain_list_of_filtered_patents", tag="answer_generation")
def obtain_list_of_filtered_patents(state: GraphState, writer: StreamWriter):
    """
    Agentic document listing for patents.
    
    Features:
    - Dynamic count, attributes, and sorting based on query
    - Follow-up support with pagination and attribute accumulation
    - Parallel execution for performance
    - LLM-powered output formatting
    - Context management across conversation turns
    - Patent-specific classification extraction and entity matching
    
    Args:
        state: GraphState containing question and other data
        writer: StreamWriter for progress messages
        
    Returns:
        Dictionary with generation and messages
    """
    
    # Check feature flag
    if not Config.USE_AGENTIC_LISTING:
        print("⚠️  Agentic listing disabled - using legacy implementation")
        return obtain_list_of_filtered_patents_legacy(state, writer)
    
    # Execute common workflow with patent-specific filter function
    return execute_listing_workflow(
        state=state,
        writer=writer,
        corpus_key="patents",
        filter_fetch_function=generate_filter_and_fetch_documents_patents
    )


@timed_step("obtain_list_of_filtered_literature", tag="answer_generation")
def obtain_list_of_filtered_literature(state: GraphState, writer: StreamWriter):
    """
    Agentic document listing for literature/papers.
    
    Features:
    - Dynamic count, attributes, and sorting based on query
    - Follow-up support with pagination and attribute accumulation
    - Parallel execution for performance
    - LLM-powered output formatting
    - Context management across conversation turns
    
    Args:
        state: GraphState containing question and other data
        writer: StreamWriter for progress messages
        
    Returns:
        Dictionary with generation and messages
    """
    
    # Check feature flag
    if not Config.USE_AGENTIC_LISTING:
        print("⚠️  Agentic listing disabled - using legacy implementation")
        # TODO: Implement legacy fallback for literature
        return {
            "generation": "Literature listing not yet implemented in legacy mode.",
            "messages": []
        }
    
    # Execute common workflow with literature-specific filter function
    return execute_listing_workflow(
        state=state,
        writer=writer,
        corpus_key="literature",
        filter_fetch_function=generate_filter_and_fetch_documents_literature
    )


@timed_step("obtain_list_of_filtered_tds", tag="answer_generation")
def obtain_list_of_filtered_tds(state: GraphState, writer: StreamWriter):
    """
    Agentic document listing for technical datasheets (TDS).
    
    Features:
    - Dynamic count, attributes, and sorting based on query
    - Follow-up support with pagination and attribute accumulation
    - Parallel execution for performance
    - LLM-powered output formatting
    - Context management across conversation turns
    
    Args:
        state: GraphState containing question and other data
        writer: StreamWriter for progress messages
        
    Returns:
        Dictionary with generation and messages
    """
    
    # Check feature flag
    if not Config.USE_AGENTIC_LISTING:
        print("⚠️  Agentic listing disabled - using legacy implementation")
        # TODO: Implement legacy fallback for TDS
        return {
            "generation": "TDS listing not yet implemented in legacy mode.",
            "messages": []
        }
    
    # Execute common workflow with TDS-specific filter function
    return execute_listing_workflow(
        state=state,
        writer=writer,
        corpus_key="tds",
        filter_fetch_function=generate_filter_and_fetch_documents_tds
    )





def decide_tds_database(state, writer: StreamWriter):
    """
    Routes to TDS database for retrieval.

    What this function does:
        Analyzes the user query and determines the appropriate technical data sheet
        (TDS) database classification for retrieval.

    Variables:
        - question: The user query to analyze.
        - data_source: The determined database classification (typically "Generic" for TDS).

    Logic:
        1. Logs the start of the decision process
        2. Extracts the user query from the state
        3. Invokes the tds_question_router to classify the query
        4. Logs the selected TDS database classification
        5. Returns the updated state with data_source and corpus_type

    Returns:
        dict: Updated state with data_source set to the appropriate classification
              and corpus_type set to "tds".
    """
    writer("Checking technical datasheets ...")
    print("--" * 10, "Deciding my knowledge source", "--" * 10)
    question = state["question"]
    data_source = tds_question_router.invoke(question)
    print(
        f"{'--' * 10} Retrieving from {data_source['datasource']} TDS database {'--' * 10}"
    )
    return {"data_source": data_source["datasource"], "corpus_type": "tds"}


def route_to_tds_database(state, writer: StreamWriter):
    """Retrieve documents from TDS data source.

    What this function does:
        Fetches relevant technical data sheet (TDS) documents based on the
        data source determined in the previous step.

    Variables:
        - source: The specific TDS database classification to query.
        - question: The user query to search with.
        - corpus_key: The type of corpus being queried (set to "tds").
        - retriever: The retriever instance configured for this query.
        - documents: The retrieved document chunks.

    Logic:
        1. Extracts the data source and query from the state
        2. Logs the start of document retrieval
        3. Checks if the source is valid for the TDS collection:
           - If valid, attempts to retrieve documents:
             * Creates an appropriate retriever with reranking if configured
             * Invokes the retriever with the query
             * Logs the number of retrieved documents
             * Returns the updated state with documents
           - If retrieval fails, returns an error message
           - If the source is invalid, returns a "learning this domain" message

    Returns:
        dict: Updated state with either:
              - The retrieved documents added to the "documents" field, or
              - An error message in the "generation" field and empty documents list
    """
    writer("Using technical datasheets as knowledge source ...")
    source = state["data_source"]
    question = state["question"]
    corpus_key = "tds"

    print(f"{'--' * 10} Obtaining documents relevant to question {'--' * 10}")

    if source in COLLECTIONS_CONFIG()[corpus_key]["classifications"]:
        try:
            # Get appropriate retriever based on configuration
            retriever = get_retriever(
                state,
                collection_type=corpus_key,
                classification=source,
                query=question,
                use_reranking=Config.USE_RERANKING,
            )

            start = time.perf_counter()
            documents = retriever.invoke(question)
            end = time.perf_counter()
            duration = end - start
            timings = state.get("timings")
            timings["retriever"] = {
                "duration_sec": round(duration, 6),
                "start_time": start,
                "end_time": end,
                "tag": "reranked retrieval",
            }
            state["timings"] = timings

            print(
                f"---RETRIEVED {len(documents)} DOCUMENT CHUNKS FROM TDS COLLECTION---"
            )
            return {"documents": documents}

        except Exception as e:
            print(f"Error during TDS retrieval: {e.__class__.__name__}")
            print(f"Error message: {str(e)}")
            return {
                "documents": [],
                "generation": "I apologize, but I encountered an error while obtaining documents relevant to your query. Please try again or rephrase your question.",
            }
    else:
        print("I am learning this domain.")
        return {"documents": [], "generation": "I am learning this domain."}


def decide_literature_database(state, writer: StreamWriter):
    """
    Routes to proper literature database for retrieval.

    What this function does:
        Analyzes the user query and determines which literature database classification
        is most appropriate for finding relevant information.

    Variables:
        - question: The user query to analyze.
        - data_source: The determined database classification.

    Logic:
        1. Logs the start of the decision process
        2. Extracts the user query from the state
        3. Invokes the literature_question_router to classify the query into one of:
           - "FRPolymers", "Elastomers", "Composites", or "Unknown"
        4. Logs the selected database classification
        5. Returns the updated state with data_source and corpus_type

    Returns:
        dict: Updated state with data_source set to the appropriate database classification
              and corpus_type set to "literature".
    """
    writer("Looking through scientific literature")
    print("--" * 10, "Deciding my knowledge source", "--" * 10)
    question = state["question"]
    data_source = literature_question_router.invoke(question)
    print(
        f"{'--' * 10} Retrieving from {data_source['datasource']} database {'--' * 10}"
    )
    return {"data_source": data_source["datasource"], "corpus_type": "literature"}


def route_to_literature_database(state, writer: StreamWriter):
    """Retrieve documents from relevant literature data source.

    What this function does:
        Fetches relevant documents from the appropriate literature database based on
        the data source determined in the previous step.

    Variables:
        - source: The specific literature database classification to query.
        - question: The user query to search with.
        - corpus_key: The type of corpus being queried (set to "literature").
        - retriever: The retriever instance configured for this query.
        - documents: The retrieved document chunks.

    Logic:
        1. Extracts the data source and query from the state
        2. Logs the start of document retrieval
        3. Checks if the source is valid for the literature collection:
           - If valid, attempts to retrieve documents:
             * Creates an appropriate retriever with reranking if configured
             * Invokes the retriever with the query
             * Logs the number of retrieved documents
             * Returns the updated state with documents
           - If retrieval fails, returns an error message
           - If the source is invalid, returns a "learning this domain" message

    Returns:
        dict: Updated state with either:
              - The retrieved documents added to the "documents" field, or
              - An error message in the "generation" field and empty documents list
    """
    writer("Using literature database as knowledge source")
    source = state["data_source"]
    question = state["question"]
    corpus_key = "literature"

    print(f"{'--' * 10} Obtaining documents relevant to question {'--' * 10}")

    if source in COLLECTIONS_CONFIG()[corpus_key]["classifications"]:
        try:
            # Get appropriate retriever based on configuration
            retriever = get_retriever(
                state,
                collection_type=corpus_key,
                classification=source,
                query=question,
                use_reranking=Config.USE_RERANKING,
            )

            start = time.perf_counter()
            documents = retriever.invoke(question)
            end = time.perf_counter()
            duration = end - start
            timings = state.get("timings")
            timings["retriever"] = {
                "duration_sec": round(duration, 6),
                "start_time": start,
                "end_time": end,
                "tag": "reranked retrieval",
            }
            state["timings"] = timings

            print(
                f"---RETRIEVED {len(documents)} DOCUMENT CHUNKS FROM LITERATURE COLLECTION---"
            )
            return {"documents": documents}

        except Exception as e:
            print(f"Error during LITERATURE retrieval: {e.__class__.__name__}")
            print(f"Error message: {str(e)}")
            return {
                "documents": [],
                "generation": "I apologize, but I encountered an error while obtaining documents relevant to your query. Please try again or rephrase your question.",
            }
    else:
        print("I am learning this domain.")
        return {"documents": [], "generation": "I am learning this domain."}


def decide_knowledge_base_database(state, writer: StreamWriter):
    """
    Routes to proper knowledge base database for retrieval.

    What this function does:
        Analyzes the user query and determines which knowledge base database classification
        is most appropriate for finding relevant information.

    Variables:
        - question: The user query to analyze.
        - data_source: The determined database classification.

    Logic:
        1. Logs the start of the decision process
        2. Extracts the user query from the state
        3. Logs the selected database classification
        4. Returns the updated state with data_source and corpus_type

    Returns:
        dict: Updated state with data_source set to the appropriate database classification
              and corpus_type set to "knowledge_base".
    """
    writer("Looking through scientific knowledge base")
    print("--" * 10, "Deciding my knowledge source", "--" * 10)
    question = state["question"]
    data_source_value = "Generic"
    return {"data_source": data_source_value, "corpus_type": "knowledge_base"}


def route_to_knowledge_base_database(state, writer: StreamWriter):
    """Retrieve documents from knowledge base data source.

    What this function does:
        Fetches relevant knowledge base documents based on the
        data source determined in the previous step.

    Variables:
        - source: The specific knowledge base classification to query.
        - question: The user query to search with.
        - corpus_key: The type of corpus being queried (set to "knowledge_base").
        - retriever: The retriever instance configured for this query.
        - documents: The retrieved document chunks.

    Logic:
        1. Extracts the data source and query from the state
        2. Logs the start of document retrieval
        3. Checks if the source is valid for the Knowledge Base collection:
           - If valid, attempts to retrieve documents:
             * Creates an appropriate retriever with reranking if configured
             * Invokes the retriever with the query
             * Logs the number of retrieved documents
             * Returns the updated state with documents
           - If retrieval fails, returns an error message
           - If the source is invalid, returns a "learning this domain" message

    Returns:
        dict: Updated state with either:
              - The retrieved documents added to the "documents" field, or
              - An error message in the "generation" field and empty documents list
    """
    writer("Using Knowledge Base as source ...")
    source = state["data_source"]
    question = state["question"]
    corpus_key = "knowledge_base"

    print(f"{'--' * 10} Obtaining documents relevant to question {'--' * 10}")

    if source in COLLECTIONS_CONFIG()[corpus_key]["classifications"]:
        try:
            # Get appropriate retriever based on configuration
            retriever = get_retriever(
                state,
                collection_type=corpus_key,
                classification=source,
                query=question,
                use_reranking=Config.USE_RERANKING,
            )

            start = time.perf_counter()
            documents = retriever.invoke(question)
            end = time.perf_counter()
            duration = end - start
            timings = state.get("timings")
            timings["retriever"] = {
                "duration_sec": round(duration, 6),
                "start_time": start,
                "end_time": end,
                "tag": "reranked retrieval",
            }
            state["timings"] = timings

            print(
                f"---RETRIEVED {len(documents)} DOCUMENT CHUNKS FROM KNOWLEDGE BASE COLLECTION---"
            )
            return {"documents": documents}

        except Exception as e:
            print(f"Error during KNOWLEDGE BASE retrieval: {e.__class__.__name__}")
            print(f"Error message: {str(e)}")
            return {
                "documents": [],
                "generation": "I apologize, but I encountered an error while obtaining documents relevant to your query. Please try again or rephrase your question.",
            }
    else:
        print("I am learning this domain.")
        return {"documents": [], "generation": "I am learning this domain."}


def grade_documents(state, writer: StreamWriter):
    """
    Filters the documents relevant to the question.

    What this function does:
        Processes and filters retrieved documents, collecting metadata and
        tracking document sources for citation purposes.

    Variables:
        - question: The user query to evaluate document relevance against.
        - documents: The list of retrieved documents to filter.
        - corpus_type: The type of corpus being queried (patent, literature, or tds).
        - filtered_docs: List to store documents after filtering.
        - document_sources: List to track source identifiers for citation.
        - document_id: Counter for document numbering in logs.
        - unique_sources: Deduplicated list of document sources.

    Logic:
        1. Extracts the question, documents, and corpus type from the state
        2. Initializes empty lists for filtered documents and sources
        3. Iterates through each document:
           - Note: Original implementation included relevance scoring, but current
             version includes all documents by default
           - Adds each document to the filtered list
           - Logs document metadata relevant to the corpus type
           - Extracts and stores the document source
        4. Deduplicates the list of sources
        5. Logs the number of unique sources and document counts
        6. Returns the updated state with filtered documents and source list

    Returns:
        dict: Updated state with filtered_docs and document_sources fields.
    """
    writer("Finding relevant documents ...")
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    corpus_type = state["corpus_type"]

    # Score each doc
    filtered_docs = []
    document_sources = []
    for document_id, document in enumerate(documents, 1):
        # NOTE: The original code included a separate retrieval grading step, but it has been commented out since a reranking step is already applied as part of the retriever, and because this step took very long and was not perfectly accurate.
        # score = retrieval_grader.invoke(
        #     {"question": question, "document": document.page_content}
        # )
        # grade = score["score"]
        # # Document relevant
        # if grade.lower() == "yes":
        #     print("---GRADE: DOCUMENT RELEVANT---")
        #     # print(f"The document contains the following information:\n{document}\n")
        filtered_docs.append(document)
        # print(f"\n---Filtered chunk {document_id}:---")
        if corpus_type == "patent":
            print(
                f"-- Filtered chunk {document_id}\nMetadata: Patent ID - {document.metadata['patent']}; Classification - {document.metadata['classifications']}; Chunk ID (pk): {document.metadata['pk']}; Relevance Score: {document.metadata.get('relevance_score', '-')}"
            )
            print(f"Content:\n{document.page_content[:200]}...")
        elif corpus_type == "literature":
            print(
                f"-- Filtered chunk {document_id}\nMetadata: Chemrxiv ID - {document.metadata['chemrxiv_id']}; Title - {document.metadata['title']}; Categories - {document.metadata['categories']}; Keywords - {document.metadata['keywords']}; Chunk ID (pk): {document.metadata['pk']}; Relevance Score: {document.metadata.get('relevance_score', '-')}"
            )
            print(f"Content:\n{document.page_content[:200]}...")
        else:
            print(f"-- Filtered document {document_id}\nMetadata:\n{document.metadata}")
            print(f"Content:\n{document.page_content[:200]}...")
        document_sources.append(document.metadata["source"])

    unique_sources = list(set(document_sources))
    print(f"\n\nUnique sources: {unique_sources}")
    print(
        f"Number of filtered documents: {len(filtered_docs)}; Total documents: {len(documents)}"
    )

    return {
        "filtered_docs": filtered_docs,
        "document_sources": unique_sources,
    }


@timed_step("generate_rag_answer", tag="answer_generation")
def generate(state):
    """
    Generate answer using RAG on retrieved documents.

    What this function does:
        Produces an answer to the user's query using Retrieval-Augmented Generation (RAG)
        with the filtered documents.

    Variables:
        - question: The user query to answer.
        - filtered_docs: The list of filtered documents to use as context.
        - document_sources: List of document source identifiers for citation.
        - generation: The generated answer text.

    Logic:
        1. Logs the start of the answer generation process
        2. Extracts the question, filtered documents, and document sources from the state
        3. Invokes the RAG chain (rag_chainA) with:
           - The filtered documents as context
           - The user query
           - The document sources for citation
        4. Returns the updated state with the generated answer and as an AI message

    Returns:
        dict: Updated state with the generation field containing the generated answer
              and messages field with the answer as an AIMessage.
    """
    print("---GENERATING ANSWER USING RAG")
    question = state["question"]
    filtered_docs = state["filtered_docs"]
    # documents = state["documents"]

    document_sources = state["document_sources"]

    # RAG generation
    generation = rag_chainA.invoke(
        {
            "context": filtered_docs,
            "question": question,
            "sources": document_sources,
        }
    )

    return {
        "generation": generation,
        "messages": [AIMessage(content=generation)],
    }


def obtain_patent_source_information(chunk_ids, retrieved_documents) -> str:
    """
    Creates a markdown formatted string containing source information for patent documents.

    What this function does:
        Creates a formatted citation block for patent documents used in the answer,
        including details like patent ID, title, and expiration date.

    Variables:
        - chunk_ids: List of chunk IDs used to generate the answer.
        - retrieved_documents: List of retrieved langchain Documents.
        - unique_chunk_ids: Deduplicated list of chunk IDs.
        - patent_source_information: The markdown string being built.
        - added_sources: Set to track already processed sources to avoid duplicates.
        - chunk_filepath: Path to the source file of the current chunk.
        - chunk_filename: Filename extracted from the filepath.
        - chunk_filename_without_extension: Filename without extension for Google Patents URL.
        - google_patents_url: URL to the patent on Google Patents.
        - patent_id: The patent identifier.
        - patent_title: Title of the patent.
        - patent_expiration_date: Expiration date of the patent.
        - relevance_score: Relevance score if reranking was used.

    Logic:
        1. Deduplicates the chunk IDs to avoid duplicate sources
        2. Initializes the output string and a set to track processed sources
        3. For each unique chunk ID:
           - Finds the matching document in retrieved_documents
           - Extracts metadata including filepath, patent ID, title, and expiration date
           - Checks if the source has already been added to avoid duplicates
           - If using reranking, formats the relevance score
           - Builds a formatted markdown string with the patent information
           - Adds hyperlinks to Google Patents where possible

    Returns:
        str: A markdown formatted string with source information for the patents
             used in generating the answer.
    """
    # Get unique chunk IDs to avoid duplicate sources
    unique_chunk_ids = list(set(chunk_ids))

    # Build source information string
    patent_source_information = "\n"

    # Track sources already added to avoid duplicates
    added_sources = set()

    def format_assignee_names(raw_assignee_data, label="Assignee"):
        """
        Cleans and formats assignee name data from a string-represented list.

        Args:
            raw_assignee_data (str): The raw string that looks like a list of assignee names.
            label (str): Optional label to prepend (e.g., "Assignee" or "Original Assignee").

        Returns:
            str: A formatted string like 'Assignee: Name1, Name2, Name3'
        """
        if isinstance(raw_assignee_data, str):
            try:
                # Try to parse the string into a list
                assignee_list = ast.literal_eval(raw_assignee_data)
                # Clean each entry
                cleaned_assignees = [
                    name.strip() for name in assignee_list if isinstance(name, str)
                ]
                if cleaned_assignees:
                    return f"{label}: {', '.join(cleaned_assignees)};\n"
            except (ValueError, SyntaxError):
                # Return raw string if it couldn't be parsed
                return f"{label}: {raw_assignee_data.strip()};\n"
        elif isinstance(raw_assignee_data, list):
            # Handle if it's already a list (just in case)
            cleaned_assignees = [
                name.strip() for name in raw_assignee_data if isinstance(name, str)
            ]
            if cleaned_assignees:
                return f"{label}: {', '.join(cleaned_assignees)};\n"
        elif isinstance(raw_assignee_data, str):
            return f"{label}: {raw_assignee_data.strip()};\n"

        return f"{label}: -;\n"  # Fallback for missing or unrecognized formats

    # For each unique chunk ID, find matching document and extract info
    source_number = 0
    for i, chunk_id in enumerate(unique_chunk_ids, 1):
        for doc in retrieved_documents:
            if doc.metadata.get("pk") == chunk_id:
                chunk_filepath = doc.metadata.get("source", "Unknown source")
                chunk_filename = chunk_filepath.split("/")[-1]
                chunk_filename_without_extension = chunk_filename.split(".")[0]
                google_patents_url = doc.metadata.get(
                    "url",
                    f"https://patents.google.com/patent/{chunk_filename_without_extension}",
                )
                patent_id = doc.metadata.get("patent", chunk_filename_without_extension)
                patent_title = doc.metadata.get("title", "-")
                patent_expiration_date = doc.metadata.get("expiration_date", "-")
                assignee_name_orig = doc.metadata.get("assignee_name_orig", "-")
                assignee_name_current = doc.metadata.get("assignee_name_current", "-")

                # Only add source if not already included
                if chunk_filepath not in added_sources:
                    source_number += 1
                    added_sources.add(chunk_filepath)
                    if Config.USE_RERANKING:
                        relevance_score = doc.metadata.get(
                            "relevance_score", "Not available"
                        )
                        # Format scores to 3 decimal places if they exist
                        try:
                            relevance_score = float(relevance_score)
                            relevance_score = f"{relevance_score:.3f}"
                        except (ValueError, TypeError):
                            # Keep the original value if conversion fails
                            pass

                    # Build markdown string
                    patent_source_information += f"\n*Source {source_number}*:\n"
                    patent_source_information += (
                        f"Patent ID: [{patent_id}]({google_patents_url});\n"
                    )
                    patent_source_information += f"Patent Title: {patent_title};\n"
                    patent_source_information += format_assignee_names(
                        assignee_name_current, label="Current Assignee"
                    )
                    patent_source_information += (
                        f"Patent Expiration Date: {patent_expiration_date};\n"
                    )
                    # patent_source_information += (
                    #     f"Filepath: .../{'/'.join(chunk_filepath.split('/')[-3:])}\n"
                    # )
                    # patent_source_information += f"- Chunk ID: {chunk_id}"
                    # patent_source_information += f"- Relevance score: {relevance_score}"
                break

    return patent_source_information


def obtain_tds_source_information(chunk_ids, retrieved_documents) -> str:
    """
    Creates a markdown formatted string containing source information for technical data sheet documents.

    What this function does:
        Creates a formatted citation block for technical data sheet (TDS) documents
        used in the answer, including details like product name, manufacturer, and year.

    Variables:
        - chunk_ids: List of chunk IDs used to generate the answer.
        - retrieved_documents: List of retrieved langchain Documents.
        - unique_chunk_ids: Deduplicated list of chunk IDs.
        - tds_source_information: The markdown string being built.
        - added_sources: Set to track already processed sources to avoid duplicates.
        - chunk_filepath: Path to the source file of the current chunk.
        - product_name: Name of the product from the TDS.
        - compound_name: Chemical name from the TDS.
        - manufacturer: Manufacturer name from the TDS.
        - application: Application information from the TDS.
        - year: Year of the TDS publication.

    Logic:
        1. Deduplicates the chunk IDs to avoid duplicate sources
        2. Initializes the output string and a set to track processed sources
        3. For each unique chunk ID:
           - Finds the matching document in retrieved_documents
           - Extracts metadata including filepath, product name, compound name,
             manufacturer, application, and year
           - Checks if the source has already been added to avoid duplicates
           - Builds a formatted markdown string with the TDS information
           - Conditionally includes certain fields only if they have values

    Returns:
        str: A markdown formatted string with source information for the TDS documents
             used in generating the answer.
    """
    # Get unique chunk IDs to avoid duplicate sources
    unique_chunk_ids = list(set(chunk_ids))

    # Build source information string
    tds_source_information = "\n"

    # Track sources already added to avoid duplicates
    added_sources = set()

    source_number = 0
    # For each unique chunk ID, find matching document and extract info
    for i, chunk_id in enumerate(unique_chunk_ids, 1):
        for doc in retrieved_documents:
            if doc.metadata.get("pk") == chunk_id:
                chunk_filepath = doc.metadata.get("source", "Unknown source")
                product_name = doc.metadata.get("commercial_product_name", "-")
                compound_name = doc.metadata.get("general_compound_name", "-")
                manufacturer = doc.metadata.get("manufacturer_company_name", "-")
                application = doc.metadata.get("application", "-")
                year = doc.metadata.get("year", "-")

                # Only add source if not already included
                if chunk_filepath not in added_sources:
                    source_number += 1
                    added_sources.add(chunk_filepath)

                    # Build markdown string
                    tds_source_information += f"\n*Source {source_number}*:\n"
                    tds_source_information += f"Product: {product_name};\n"

                    if compound_name != "-":
                        tds_source_information += f"Type: {compound_name};\n"
                    tds_source_information += f"Manufacturer: {manufacturer};\n"
                    if application != "-":
                        tds_source_information += f"Application: {application};\n"
                    tds_source_information += f"Year: {year};\n"
                break

    return tds_source_information


def obtain_literature_source_information(chunk_ids, retrieved_documents) -> str:
    """
    Creates a markdown formatted string containing source information for literature documents.

    What this function does:
        Creates a formatted citation block for scientific literature documents used
        in the answer, including details like paper title, authors, and publication date.

    Variables:
        - chunk_ids: List of chunk IDs used to generate the answer.
        - retrieved_documents: List of retrieved langchain Documents.
        - unique_chunk_ids: Deduplicated list of chunk IDs.
        - literature_source_information: The markdown string being built.
        - added_sources: Set to track already processed sources to avoid duplicates.
        - chunk_filepath: Path to the source file of the current chunk.
        - chunk_filename: Filename extracted from the filepath.
        - chunk_filename_without_extension: Filename without extension.
        - paper_title: Title of the paper.
        - chemrxiv_id: ChemRxiv identifier if available.
        - file_pdf_url: URL to the PDF file if available.
        - authors: List of authors of the paper.
        - published_date: Publication date of the paper.
        - authors_str: Comma-separated string of author names.

    Logic:
        1. Deduplicates the chunk IDs to avoid duplicate sources
        2. Initializes the output string and a set to track processed sources
        3. For each unique chunk ID:
           - Finds the matching document in retrieved_documents
           - Extracts metadata including filepath, paper title, authors, and dates
           - Checks if the source has already been added to avoid duplicates
           - Handles author lists by converting to comma-separated string if needed
           - Builds a formatted markdown string with the literature information
           - Adds hyperlinks to ChemRxiv and PDF files when available

    Returns:
        str: A markdown formatted string with source information for the literature
             documents used in generating the answer.
    """
    # Get unique chunk IDs to avoid duplicate sources
    unique_chunk_ids = list(set(chunk_ids))

    # Build source information string
    literature_source_information = "\n"

    # Track sources already added to avoid duplicates
    added_sources = set()

    # For each unique chunk ID, find matching document and extract info
    source_number = 0
    for i, chunk_id in enumerate(unique_chunk_ids, 1):
        for doc in retrieved_documents:
            if doc.metadata.get("pk") == chunk_id:
                chunk_filepath = doc.metadata.get("source", "Unknown source")
                chunk_filename = chunk_filepath.split("/")[-1]
                chunk_filename_without_extension = chunk_filename.split(".")[0]
                paper_title = doc.metadata.get("title", "Unknown Title")
                chemrxiv_id = doc.metadata.get("chemrxiv_id", "")
                file_pdf_url = doc.metadata.get("file_pdf_url", "")
                authors = doc.metadata.get("authors", [])
                published_date = doc.metadata.get("published_time", "-")
                published_year = doc.metadata.get("published_year", "-")
                published_month = doc.metadata.get("published_month", "-")
                published_day = doc.metadata.get("published_day", "-")

                # Handle authors list
                authors_str = (
                    ", ".join(authors) if isinstance(authors, list) else authors
                )

                # Only add source if not already included
                if chunk_filepath not in added_sources:
                    source_number += 1
                    added_sources.add(chunk_filepath)

                    # Build markdown string with hyperlinks
                    literature_source_information += f"\n*Source {source_number}*:\n"

                    # Add title with ChemRxiv link if ID exists
                    if chemrxiv_id and chemrxiv_id != "NA":
                        literature_source_information += f"Paper Title: [{paper_title}](https://chemrxiv.org/engage/chemrxiv/article-details/{chemrxiv_id});\n"
                    else:
                        literature_source_information += (
                            f"Paper Title: {paper_title};\n"
                        )

                    # Add File PDF link if URL exists
                    if file_pdf_url and file_pdf_url != "NA":
                        literature_source_information += (
                            f"[File PDF]({file_pdf_url});\n"
                        )

                    # Add authors and publication date
                    if authors_str:
                        literature_source_information += f"Authors: {authors_str};\n"
                    if published_date != "-":
                        if (
                            published_year != "-"
                            and published_month != "-"
                            and published_day != "-"
                        ):
                            literature_source_information += f"Publication Date: {published_year}-{published_month}-{published_day};\n"
                break

    return literature_source_information


def obtain_knowledge_base_source_information(chunk_ids, retrieved_documents) -> str:
    """
    Creates a markdown formatted string containing source information for knowledge base documents.

    What this function does:
        Creates a formatted citation block for knowledge base documents used
        in the answer, including details like title and url.

    Variables:
        - chunk_ids: List of chunk IDs used to generate the answer.
        - retrieved_documents: List of retrieved langchain Documents.
        - unique_chunk_ids: Deduplicated list of chunk IDs.
        - knowledge_base_source_information: The markdown string being built.
        - added_sources: Set to track already processed sources to avoid duplicates.
        - chunk_filepath: Path to the source file of the current chunk.
        - chunk_filename: Filename extracted from the filepath.
        - chunk_filename_without_extension: Filename without extension.
        - title: Title of the document.
        - url: URL to the document if available.

    Logic:
        1. Deduplicates the chunk IDs to avoid duplicate sources
        2. Initializes the output string and a set to track processed sources
        3. For each unique chunk ID:
           - Finds the matching document in retrieved_documents
           - Extracts metadata including title and url
           - Checks if the source has already been added to avoid duplicates
           - Builds a formatted markdown string with the knowledge base information
           - Adds hyperlinks to the url when available

    Returns:
        str: A markdown formatted string with source information for the knowledge base
             documents used in generating the answer.
    """
    # Get unique chunk IDs to avoid duplicate sources
    unique_chunk_ids = list(set(chunk_ids))

    # Build source information string
    knowledge_base_source_information = "\n"

    # Track sources already added to avoid duplicates
    added_sources = set()

    # For each unique chunk ID, find matching document and extract info
    source_number = 0
    for i, chunk_id in enumerate(unique_chunk_ids, 1):
        for doc in retrieved_documents:
            if doc.metadata.get("pk") == chunk_id:
                chunk_filepath = doc.metadata.get("source", "Unknown source")
                chunk_filename = chunk_filepath.split("/")[-1]
                chunk_filename_without_extension = chunk_filename.split(".")[0]
                title = doc.metadata.get("title", "")
                url = doc.metadata.get("url", "")
                patent_id = doc.metadata.get("patent", "")
                patent_expiration_date = doc.metadata.get("expiration_date", "")
                assignee_name_current = doc.metadata.get("assignee_name_current", "")

                # Only add source if not already included
                if chunk_filepath not in added_sources:
                    source_number += 1
                    added_sources.add(chunk_filepath)

                    # Build markdown string with hyperlinks
                    knowledge_base_source_information += f"\n*Source {source_number}*: "

                    # Add URL if it exists
                    if patent_id and patent_id != "":
                        if url and url != "NA":
                            if title and title != "":
                                knowledge_base_source_information += (
                                    f"[{patent_id} - {title}]({url}) "
                                )
                            else:
                                knowledge_base_source_information += (
                                    f"[{patent_id}]({url});\n"
                                )
                        else:
                            knowledge_base_source_information += (
                                f"{patent_id} - {title};\n"
                            )
                    elif title and title != "":
                        if url and url != "NA":
                            knowledge_base_source_information += f"[{title}]({url});\n"
                        else:
                            knowledge_base_source_information += f"{title};\n"
                    else:
                        knowledge_base_source_information += (
                            f"{chunk_filename_without_extension};\n"
                        )

                    if assignee_name_current and assignee_name_current != "":
                        knowledge_base_source_information += (
                            f"Assignee: {assignee_name_current};\n"
                        )
                    if patent_expiration_date and patent_expiration_date != "":
                        knowledge_base_source_information += (
                            f"Expiration Date: {patent_expiration_date};\n"
                        )

                break

    return knowledge_base_source_information


@timed_step("generate_with_formatted_sources", tag="answer_generation")
def generate_with_formatted_sources(state):
    """
    Generate answer using RAG on retrieved documents with formatted source citations.

    What this function does:
        Produces an answer to the user's query using Retrieval-Augmented Generation (RAG)
        with properly formatted source citations based on the document type.

    Variables:
        - question: The user query to answer.
        - filtered_docs: The list of filtered documents to use as context.
        - corpus_type: The type of corpus being queried (patent, literature, or tds).
        - context: The formatted context string created from document joining.
        - generation_json: The raw JSON response from the RAG chain.
        - llm_answer: The answer text extracted from the response.
        - chunk_ids: List of chunk IDs used to generate the answer.
        - formatted_sources_information: Formatted citation block for sources.
        - generation: The final answer with source citations appended.

    Logic:
        1. Logs the start of the answer generation process
        2. Extracts the question, filtered documents, and corpus type from the state
        3. Checks if there are documents to work with:
           - If no documents, returns an error message
        4. Joins the documents based on corpus type using the appropriate function
        5. Generates an answer with the RAG chain:
           - Handles potential errors during generation
        6. Extracts the answer and chunk IDs from the JSON response:
           - Validates the response format and content
        7. Gets formatted source information based on corpus type:
           - Uses corpus-specific functions to format citations
           - Handles potential errors in source formatting
        8. Combines the answer with the formatted source information
        9. Returns the updated state with the generated answer and as an AI message

    Returns:
        dict: Updated state with the generation field containing the generated answer
              with formatted sources, and messages field with the answer as an AIMessage.
    """
    try:
        print("---GENERATING ANSWER USING RAG WITH FORMATTED SOURCES---")
        question = state["question"]
        filtered_docs = state["filtered_docs"]
        corpus_type = state["corpus_type"]

        # Check if we have any documents to work with
        if not filtered_docs:
            error_message = "I apologize, but I couldn't find any relevant documents to answer your question. Could you please rephrase your question or provide more details?"
            return {
                "generation": error_message,
                "messages": [AIMessage(content=error_message)],
            }

        # Join documents based on corpus type
        if corpus_type == "patent":
            context = join_patent_documents(filtered_docs)
        elif corpus_type == "literature":
            context = join_literature_documents(filtered_docs)
        elif corpus_type == "tds":
            context = join_tds_documents(filtered_docs)
        elif corpus_type == "knowledge_base":
            context = join_knowledge_base_documents(filtered_docs)
        else:
            raise ValueError(f"Unknown corpus type: {corpus_type}")

        # RAG generation with timeout handling
        try:
            generation_json = rag_chainC.invoke(
                {
                    "context": context,
                    "question": question,
                },
            )
        except Exception as e:
            error_message = "I apologize, but I encountered an error while generating the answer. Please try again or rephrase your question."
            print(f"Error generating answer:\n{str(e)}")
            return {
                "generation": error_message,
                "messages": [AIMessage(content=error_message)],
            }

        print("---ANSWER GENERATED. OBTAINING SOURCES---")
        print(f"Generation JSON: {generation_json}")

        # Validate generation_json structure
        if not isinstance(generation_json, dict):
            raise ValueError("Generation result is not in the expected JSON format")

        # Extract answer and chunk_ids with fallbacks
        llm_answer = generation_json.get("answer", "")
        if not llm_answer:
            raise ValueError("No answer was generated")

        chunk_ids = generation_json.get("chunk_ids", [])
        # If chunk_ids is not a list or is None, initialize as empty list
        if not isinstance(chunk_ids, list):
            chunk_ids = []

        # Get formatted source information
        formatted_sources_information = ""
        try:
            if corpus_type == "patent":
                formatted_sources_information = obtain_patent_source_information(
                    chunk_ids, filtered_docs
                )
            elif corpus_type == "literature":
                formatted_sources_information = obtain_literature_source_information(
                    chunk_ids, filtered_docs
                )
            elif corpus_type == "tds":
                formatted_sources_information = obtain_tds_source_information(
                    chunk_ids, filtered_docs
                )
            elif corpus_type == "knowledge_base":
                formatted_sources_information = (
                    obtain_knowledge_base_source_information(chunk_ids, filtered_docs)
                )
        except Exception as e:
            print(f"Error formatting sources: {str(e)}")
            formatted_sources_information = (
                "\n\nNote: Source information could not be retrieved."
            )

        print("---COMBINING ANSWER & SOURCES. PRINTING...---")
        generation = llm_answer + formatted_sources_information
        print(f"Answer:\n\n{generation}\n")
        print("---DONE!---")

        return {
            "generation": generation,
            "messages": [AIMessage(content=generation)],
        }

    except Exception as e:
        print(f"Error in generate_with_formatted_sources: {str(e)}")
        error_message = "I apologize, but I encountered an error while generating the answer. Please try again or rephrase your question."
        return {
            "generation": error_message,
            "messages": [AIMessage(content=error_message)],
        }


@timed_step("generate_summary_with_formatted_sources", tag="answer_generation")
def generate_summary_with_formatted_sources(state):
    try:
        print("---GENERATING SUMMARY USING RAG WITH FORMATTED SOURCES---")
        question = state["question"]
        filtered_docs = state["filtered_docs"]
        corpus_type = state["corpus_type"]

        # Check if we have any documents to work with
        if not filtered_docs:
            error_message = "I apologize, but I couldn't find any relevant documents to answer your question. Could you please rephrase your question or provide more details?"
            return {
                "generation": error_message,
                "messages": [AIMessage(content=error_message)],
            }

        # Join documents based on corpus type
        if corpus_type == "patent":
            context = join_patent_documents(filtered_docs)
        elif corpus_type == "literature":
            context = join_literature_documents(filtered_docs)
        elif corpus_type == "tds":
            context = join_tds_documents(filtered_docs)
        else:
            raise ValueError(f"Unknown corpus type: {corpus_type}")

        # RAG generation with timeout handling
        try:
            if corpus_type == "patent":
                generation_json = patent_summary_chain.invoke(
                    {
                        "context": context,
                        "question": question,
                    },
                )
            else:
                generation_json = {
                    "answer": f"I am not able to summarize a {corpus_type} document since a summary template has not been created yet.",
                    "chunk_ids": [],
                }

        except Exception as e:
            error_message = "I apologize, but I encountered an error while generating the summary. Please try again or rephrase your question."
            print(f"Error generating answer{e.__class__.__name__}:\n{str(e)}")
            return {
                "generation": error_message,
                "messages": [AIMessage(content=error_message)],
            }

        print("---SUMMARY GENERATED. OBTAINING SOURCES---")
        print(f"Generation JSON: {generation_json}")

        # Validate generation_json structure
        if not isinstance(generation_json, dict):
            raise ValueError("Generation result is not in the expected JSON format")

        # Extract answer and chunk_ids with fallbacks
        llm_answer = generation_json.get("answer", "")
        if not llm_answer:
            raise ValueError("No answer was generated")

        chunk_ids = generation_json.get("chunk_ids", [])
        # If chunk_ids is not a list or is None, initialize as empty list
        if not isinstance(chunk_ids, list):
            chunk_ids = []

        print(f"Chunk IDs: {chunk_ids}")

        # Get formatted source information
        formatted_sources_information = ""
        try:
            if corpus_type == "patent":
                formatted_sources_information = obtain_patent_source_information(
                    chunk_ids, filtered_docs
                )
            elif corpus_type == "literature":
                formatted_sources_information = obtain_literature_source_information(
                    chunk_ids, filtered_docs
                )
            elif corpus_type == "tds":
                formatted_sources_information = obtain_tds_source_information(
                    chunk_ids, filtered_docs
                )
        except Exception as e:
            print(f"Error formatting sources: {str(e)}")
            formatted_sources_information = (
                "\n\nNote: Source information could not be retrieved."
            )

        print("---COMBINING SUMMARY & SOURCES. PRINTING...---")
        generation = llm_answer + formatted_sources_information
        return {
            "generation": generation,
            "messages": [AIMessage(content=generation)],
        }

    except Exception as e:
        print(f"Error in generate_summary_with_formatted_sources: {str(e)}")
        error_message = "I apologize, but I encountered an error while generating the summary. Please try again or rephrase your question."
        return {
            "generation": error_message,
            "messages": [AIMessage(content=error_message)],
        }


def generate_chem(state):
    """
    Generate answer using LLM for chemistry-related queries.

    What this function does:
        Produces an answer to a chemistry question using only the LLM's knowledge,
        without retrieval-augmented generation.

    Variables:
        - question: The user query to answer.
        - generation_chain: Chain combining the generation prompt with the chat model.
        - generation: The generated answer text.

    Logic:
        1. Logs the start of the answer generation process
        2. Extracts the question from the state
        3. Creates a generation chain combining the template with the chat model
        4. Generates an answer using only the LLM knowledge
        5. Returns the updated state with the generated answer, preserved question,
           and the answer as an AI message

    Returns:
        dict: Updated state with question, generation, and messages fields.
    """
    # NOTE: This node is not being used currently
    print("---GENERATING ANSWER USING LLM---")
    question = state["question"]

    # Generation
    generation_chain = (
        generation_prompt | chat_model | gen_int_out_logger | StrOutputParser()
    )
    generation = generation_chain.invoke(question)
    return {
        "question": question,
        "generation": generation,
        "messages": [AIMessage(content=generation)],
    }


def answer_greeting_query(state):
    """
    Answer the user query when it is a simple greeting.

    What this function does:
        Provides a standard response to greeting messages like "hello".

    Variables:
        - greeting_query_answer: The standard greeting response message.

    Logic:
        1. Logs the answering of a greeting query
        2. Creates a standard greeting response
        3. Returns the updated state with the greeting response as the generation
           and as an AI message

    Returns:
        dict: Updated state with generation and messages fields containing the
              standard greeting response.
    """
    print("---ANSWERING GREETING QUERY---")
    greeting_query_answer = (
        "Hello! I am Simreka AI. Can I assist you with any chemistry-related questions?"
    )
    return {
        # "question": question,
        "generation": greeting_query_answer,
        "messages": [AIMessage(content=greeting_query_answer)],
    }


def answer_with_standard_response(state):
    """
    Standard response for queries not related to chemistry.

    What this function does:
        Provides a standard response when the user asks about non-chemistry topics,
        indicating the system's domain limitations.

    Variables:
        - standard_response: The standard response message for non-chemistry queries.

    Logic:
        1. Creates a standard response message indicating the system's focus on chemistry
        2. Returns the updated state with the standard response as the generation
           and as an AI message

    Returns:
        dict: Updated state with generation and messages fields containing the
              standard non-chemistry response.
    """
    standard_response = "I am only able to answer chemistry-related questions. Please ask a question related to the domain of chemistry and I would be happy to help."
    return {
        "generation": standard_response,
        "messages": [AIMessage(content=standard_response)],
    }


def answer_llm_nature_query(state):
    """
    Answer the user query about the nature of the LLM.

    What this function does:
        Provides a standard response when the user asks about the AI system itself,
        its capabilities, or identity.

    Variables:
        - llm_nature_answer: The standard response for queries about the AI system.

    Logic:
        1. Logs the answering of an LLM nature query
        2. Creates a standard response about the system's identity
        3. Returns the updated state with the standard response as the generation
           and as an AI message

    Returns:
        dict: Updated state with generation and messages fields containing the
              standard LLM nature response.
    """
    print("---ANSWERING LLM NATURE QUERY---")
    llm_nature_answer = (
        "I am Simreka AI and I am able to answer your questions related to chemistry."
    )
    return {
        "generation": llm_nature_answer,
        "messages": [AIMessage(content=llm_nature_answer)],
    }


def generic_node(state):
    """
    Generic node that does nothing.

    What this function does:
        Acts as a placeholder node that returns the state unchanged.

    Variables:
        - state: The current graph state, passed through unchanged.

    Logic:
        Simply returns without modifying the state (pass-through function).

    Returns:
        None: Returns no updates to the state.
    """
    return


# class TokenUsageTracker:
#     """Track token usage across LLM invocations"""
#
#     def __init__(self):
#         self.usage_history = []
#         self.total_input_tokens = 0
#         self.total_output_tokens = 0
#         self.total_tokens = 0
#
#     def record_usage(self, usage_metadata: dict, context: str = ""):
#         """Record token usage from an LLM response"""
#         if usage_metadata:
#             usage_record = {
#                 "timestamp": datetime.now().isoformat(),
#                 "context": context,
#                 "input_tokens": usage_metadata.get("input_tokens", 0),
#                 "output_tokens": usage_metadata.get("output_tokens", 0),
#                 "total_tokens": usage_metadata.get("total_tokens", 0),
#             }
#
#             self.usage_history.append(usage_record)
#             self.total_input_tokens += usage_record["input_tokens"]
#             self.total_output_tokens += usage_record["output_tokens"]
#             self.total_tokens += usage_record["total_tokens"]
#
#     def print_usage(self):
#         """Print the latest token usage in a readable format"""
#         if not self.usage_history:
#             print("No token usage history available.")
#             return
#
#         last_usage = self.usage_history[-1]
#         print("\n--- Latest Token Usage ---")
#         print(f"Context: {last_usage['context']}")
#         print(f"Input tokens: {last_usage['input_tokens']:,}")
#         print(f"Output tokens: {last_usage['output_tokens']:,}")
#         print(f"Total tokens: {last_usage['total_tokens']:,}")
#         print(f"Timestamp: {last_usage['timestamp']}")
#         print("-------------------------\n")
#
#     def get_summary(self) -> dict:
#         """Get usage summary"""
#         return {
#             "total_invocations": len(self.usage_history),
#             "total_input_tokens": self.total_input_tokens,
#             "total_output_tokens": self.total_output_tokens,
#             "total_tokens": self.total_tokens,
#             "average_input_per_call": self.total_input_tokens
#             / max(len(self.usage_history), 1),
#             "average_output_per_call": self.total_output_tokens
#             / max(len(self.usage_history), 1),
#         }
#
#
# # Initialize global token tracker
# token_tracker = TokenUsageTracker()


@timed_step("gen_int_out_node", tag="llm_only_answer_generation")
def gen_int_out_node(state):
    """
    Generate an output using the generic intelligence output chain.

    What this function does:
        Processes a query using the general-purpose generation chain that handles
        non-specialized queries.

    Variables:
        - context: The conversation history for context.
        - question: The current user query.
        - gen_int_out: The generated response.

    Logic:
        1. Extracts the conversation history and current query from the state
        2. Invokes the generic intelligence output chain with the context and question
        3. Returns the updated state with the generated response

    Returns:
        dict: Updated state with the generation field containing the generic response.
    """
    # print("reached", state)
    context = state["messages"]
    question = state["messages"][-1].content

    gen_int_out = gen_int_out_chain.invoke({"context": context, "question": question})
    print(f"Generation:\n{gen_int_out}")

    return {"generation": gen_int_out}


# Conditional edge functions
def choose_answer_generation_path(state, writer: StreamWriter):
    """
    Check if the user query contains the keywords for patent, literature, or TDS.

    What this function does:
        Analyzes the user query for specific keywords to determine which database
        path to follow for document retrieval.

    Variables:
        - messages: The conversation history.
        - question: The current user query.
        - rag_keywords: Keywords that indicate a patent-related query.
        - literature_keywords: Keywords that indicate a literature-related query.
        - tds_keywords: Keywords that indicate a TDS-related query.
        - question_lower: Lowercase version of the query for case-insensitive matching.

    Logic:
        1. Extracts the latest user query from the message history
        2. Logs the query for debugging
        3. Defines keyword lists for different document types
        4. Converts the query to lowercase for case-insensitive matching
        5. Checks for TDS keywords first (priority order)
        6. Then checks for patent keywords
        7. Then checks for literature keywords
        8. If no keywords match, routes to the generic intelligence output node

    Returns:
        Command: A Command object directing the graph to the appropriate next node.
    """
    writer("Deciding how to answer your question ...")
    messages = state["messages"]
    if "document_filter" in state:
        source = state["document_filter"]
        print("Printing Source : ", source)

        # If source is specified, route directly based on document filter
        if source == "patent":
            print("Routing to PATENT DATABASE based on document filter")
            return "generation_path:patent"
        elif source == "literature":
            print("Routing to LITERATURE DATABASE based on document filter")
            return "generation_path:literature"
        elif source == "tds":
            print("Routing to TDS DATABASE based on document filter")
            return "generation_path:tds"
        # Note: "web" case should be handled by web search before reaching this point
        elif source == "web":
            print("Web source detected - this should be handled by web search")
            return "generation_path:general"
        elif source == "knowledge_base":
            print(" Routing to KNOWLEDGE BASE DATABASE based on document filter")
            return "generation_path:knowledge_base"
        elif source is None:
            print("No document filter specified - using general generation")
            return "generation_path:general"
        else:
            print(
                f"Unknown document filter: {source} - defaulting to general generation"
            )
            return "generation_path:general"

    # If no document_filter in state, fall back to keyword-based routing (legacy behavior)
    question = messages[-1].content
    print(
        f"No document filter found, using keyword-based routing for question: {question}"
    )

    rag_keywords = ["patent", "invention"]
    literature_keywords = ["literature", "research"]
    tds_keywords = ["tds", "technical datasheet"]
    knowledge_base_keywords = ["knowledge base", "internal document"]

    question_lower = question.lower()


    # Check keywords in priority order and return corresponding keys
    for keyword in knowledge_base_keywords:
        if keyword in question_lower:
            return "generation_path:knowledge_base"

    for keyword in tds_keywords:
        if keyword in question_lower:
            return "generation_path:tds"

    for keyword in rag_keywords:
        if keyword in question_lower:
            return "generation_path:patent"

    for keyword in literature_keywords:
        if keyword in question_lower:
            return "generation_path:literature"

    # Default case - route to general intelligence
    return "generation_path:general"




def decide_if_list_documents_query(state):
    """
    Decides if the user query is related to listing a set of documents.
    Now also determines which corpus type for proper routing.
    """
    start = time.perf_counter()
    
    print("=" * 10, "Checking if user query is about listing a set of document", "=" * 10)
    
    question = state["question"]
    corpus_type = state.get("corpus_type")
    
    if not corpus_type:
        print("Warning : Document Listing might fail because corpus type is not set.")
    list_documents_query_check = list_documents_query_checker.invoke({"question": question})
    is_list_documents_query = list_documents_query_check["check_flag"].lower()
    
    print(f"Query related to listing multiple documents: {is_list_documents_query}")
    
    end = time.perf_counter()
    duration = end - start
    timings = state.get("timings", {})
    timings["decide_if_list_documents_query"] = {
        "duration_sec": round(duration, 6),
        "start_time": start,
        "end_time": end,
        "tag": "query_handling",
    }
    
    if is_list_documents_query == "yes":
        # Route based on corpus type
        if corpus_type == "patent":
            return "list_documents_query_check_True_patents"
        elif corpus_type == "literature":
            return "list_documents_query_check_True_literature"
        elif corpus_type == "tds":
            return "list_documents_query_check_True_tds"
        else:
            # Fallback to patents if corpus type not set
            print(f"Warning: corpus_type not set or unknown ({corpus_type}), defaulting to patents")
            return "list_documents_query_check_True_patents"
    else:
        # Route to normal retrieval based on corpus type
        if corpus_type == "patent":
            return "list_documents_query_check_False_patents"
        elif corpus_type == "literature":
            return "list_documents_query_check_False_literature"
        elif corpus_type == "tds":
            return "list_documents_query_check_False_tds"
        else:
            # Fallback to patents if corpus type not set
            return "list_documents_query_check_False_patents"



def decide_if_summarization_needed(state):
    """
    Decides if the user query is related to summarization of a document.
    """
    start = time.perf_counter()
    print(
        "--" * 10, "Checking if user query is about summarizing a document", "--" * 10
    )
    question = state["question"]
    corpus_type = state["corpus_type"]
    summarization_check = summarization_query_checker.invoke(question)
    is_summarization_related_query = summarization_check["check_flag"].lower()
    print(f"Summarisation related query: {is_summarization_related_query}")
    end = time.perf_counter()
    duration = end - start
    timings = state.get("timings")
    timings["decide_if_summarization_needed"] = {
        "duration_sec": round(duration, 6),
        "start_time": start,
        "end_time": end,
        "tag": "query_handling",
    }
    state["timings"] = timings
    if is_summarization_related_query == "yes":
        return "summarization_needed:True"
    else:
        return "summarization_needed:False"


def decide_if_chemistry_query(state):
    """
    Decides if the user query is related to chemistry.

    What this function does:
        Determines whether the current query is related to chemistry, which affects
        the subsequent processing path.

    Variables:
        - question: The current user query.
        - chemistry_check: Result from the chemistry query checker.
        - is_chemistry_related_query: The first letter of the result, converted to lowercase.

    Logic:
        1. Logs the start of the chemistry relevance check
        2. Extracts the latest user query from the message history
        3. Invokes the chemistry query checker to analyze the query
        4. Extracts the first letter of the result and converts to lowercase
        5. Returns the appropriate edge label:
           - "chemistry_check:True" if the query is chemistry-related
           - "chemistry_check:False" if the query is not chemistry-related

    Returns:
        str: Edge label indicating whether the query is chemistry-related.
    """

    print("--" * 10, "Deciding whether user query is related to chemistry", "--" * 10)
    start = time.perf_counter()
    question = state["messages"][-1].content
    chemistry_check = chemistry_query_checker.invoke(question)
    print(chemistry_check)
    is_chemistry_related_query = chemistry_check["datasource"][0].lower()
    print(f"Chemistry-related query: {is_chemistry_related_query}")
    end = time.perf_counter()
    duration = end - start
    timings = state.get("timings")
    timings["decide_if_chemistry_query"] = {
        "duration_sec": round(duration, 6),
        "start_time": start,
        "end_time": end,
        "tag": "query_handling",
    }
    state["timings"] = timings
    # Query relevant to chemistry
    if is_chemistry_related_query == "y":
        return "chemistry_check:True"
    # Query not relevant to chemistry
    else:
        return "chemistry_check:False"


def decide_if_llm_nature_query(state):
    """
    Check if the user query is about the nature of the LLM itself.

    What this function does:
        Determines whether the current query is asking about the AI system itself,
        which affects how the query should be answered.

    Variables:
        - question: The current user query.
        - llm_nature_check: Result from the LLM nature query checker.
        - is_llm_nature_related_query: The result, converted to lowercase.

    Logic:
        1. Logs the start of the LLM nature check
        2. Extracts the query from the state
        3. Invokes the LLM nature query checker to analyze the query
        4. Extracts the result and converts to lowercase
        5. Returns the appropriate edge label:
           - "llm_query:True" if the query is about the LLM
           - "llm_query:False" if the query is not about the LLM

    Returns:
        str: Edge label indicating whether the query is about the LLM itself.
    """
    print("--" * 10, "Checking if user query is about the LLM itself", "--" * 10)
    question = state["question"]
    llm_nature_check = llm_nature_query_checker.invoke(question)
    is_llm_nature_related_query = llm_nature_check["datasource"].lower()
    print(f"LLM-Nature related query: {is_llm_nature_related_query}")
    if is_llm_nature_related_query == "yes":
        # print("I am Simreka AI and I am able to answer your questions related to chemistry.")
        return "llm_query:True"
    else:
        return "llm_query:False"


def decide_if_greeting_query(state):
    """
    Check if the user query is a simple greeting.

    What this function does:
        Determines whether the current query is just a simple greeting like "hello",
        which affects how the query should be answered.

    Variables:
        - question: The current user query.
        - greeting_check: Result from the greeting query checker.
        - is_greeting_related_query: The result, converted to lowercase.

    Logic:
        1. Extracts the query from the state
        2. Invokes the greeting query checker to analyze the query
        3. Extracts the result and converts to lowercase
        4. Returns the appropriate edge label:
           - "greeting_query:True" if the query is a greeting
           - "greeting_query:False" if the query is not a greeting

    Returns:
        str: Edge label indicating whether the query is a simple greeting.
    """
    # print("--" * 10, "Checking if user query is simply a greeting", "--" * 10)
    question = state["question"]
    greeting_check = greeting_query_checker.invoke(question)
    is_greeting_related_query = greeting_check["datasource"].lower()
    # print(f"Greeting related query: {is_greeting_related_query}")
    if is_greeting_related_query == "yes":
        # print("Hello! I am Simreka AI and I am able to answer your questions related to chemistry. How can I help?")
        return "greeting_query:True"
    else:
        return "greeting_query:False"


def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    What this function does:
        Evaluates the quality of the generated answer by checking if it's grounded
        in the retrieved documents and if it answers the user's question.

    Variables:
        - question: The user query.
        - generation: The generated answer.
        - documents: The filtered documents used for generation.
        - score: Result from the ground truth grader.
        - ground_truth_grade: Whether the answer is grounded in the documents.
        - answer_relevance_grade: Whether the answer addresses the question.

    Logic:
        1. Logs the start of the hallucination check
        2. Extracts the question, generation, and documents from the state
        3. Checks if there are filtered documents (early return if none)
        4. Checks if the generation is grounded in the documents:
           - Invokes the ground truth grader
           - If grounded, proceeds to check if it answers the question:
             * Invokes the answer relevance grader
             * If it answers the question, returns "useful"
             * Otherwise, returns "useful" anyway (with a note)
           - If not grounded, returns "not useful"

    Returns:
        str: Decision for next node to call - "useful" or "not useful".
    """
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    generation = state["generation"]
    documents = state["filtered_docs"]

    # TODO: Update this to handle empty documents
    print(f"Number of filtered documents: {len(documents)}")
    # if len(documents)==0:
    #     return "not supported"
    score = ground_truth_grader.invoke(
        {"generation": generation, "documents": documents}
    )
    ground_truth_grade = score["score"]
    # Check hallucination
    if ground_truth_grade.lower() == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_relevance_grader.invoke(
            {"question": question, "generation": generation}
        )
        answer_relevance_grade = score["score"]
        if answer_relevance_grade == "yes":
            print("---DECISION: GENERATION ANSWERS THE QUESTION---")
            return "useful"
        else:
            # TODO: Update what string is returned
            print("---DECISION: MAY NOT ANSWER YOUR QUESTION PLEASE VERIFY---")
            # user_input = input(f"Generated answer maynot answer your question. Do you want me to rerun:" )
            # if user_input.lower() == 'yes':
            #     print('rerun')
            #     return "not useful"
            # else:
            return "useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        # user_input = input(f"Generated answer is not grounded in documents, Do you want me to rerun:")
        # if user_input.lower() == 'yes':
        #     print('rerun')
        #     return "not useful"
        # else:
        return "not useful"


### Graph Construction
# Build the workflow graph with nodes and edges
workflow = StateGraph(GraphState)

# Define the nodes
# workflow.add_node("ANSWER_GENERATION_PATH", choose_answer_generation_path) # Remove node since this has become a conditional edge
workflow.add_node("GEN_INT_OUT_NODE", gen_int_out_node)
# Reformulate the user query based on existing chat history
workflow.add_node("REFORMULATE_QUESTION", reformulate_question)

# Choose appropriate patent database for retrieval based on user query
workflow.add_node("DECIDE_PATENT_DATABASE", decide_patent_database)
# Retrieve documents using embeddings and user query for patent-related queries
workflow.add_node("ROUTE_TO_PATENT_DATABASE", route_to_patent_database)
# Grade retrieved patent documents based on relevance to user query
workflow.add_node("FILTER_PATENT_DOCUMENTS", grade_documents)
# Obtain list of filtered patents if the user query is about listing documents
workflow.add_node("OBTAIN_LIST_OF_FILTERED_PATENTS", obtain_list_of_filtered_patents)
workflow.add_node("OBTAIN_LIST_OF_FILTERED_LITERATURE", obtain_list_of_filtered_literature)
workflow.add_node("OBTAIN_LIST_OF_FILTERED_TDS", obtain_list_of_filtered_tds)
# Generate answer using RAG for patent-related queries
workflow.add_node("GENERATE_PATENT_SUMMARY", generate_summary_with_formatted_sources)
if Config.USE_GENERATION_WITH_SOURCES_FOR_PATENTS:
    workflow.add_node("GENERATE_PATENT", generate_with_formatted_sources)
else:
    workflow.add_node("GENERATE_PATENT", generate)

# Choose appropriate literature database for retrieval based on user query
workflow.add_node("DECIDE_LITERATURE_DATABASE", decide_literature_database)
# Retrieve documents using embeddings and user query for literature-related queries
workflow.add_node("ROUTE_TO_LITERATURE_DATABASE", route_to_literature_database)
# Grade retrieved literature documents based on relevance to user query
workflow.add_node("FILTER_LITERATURE_DOCUMENTS", grade_documents)
# Generate answer using RAG for literature-related queries
if Config.USE_GENERATION_WITH_SOURCES_FOR_LITERATURE:
    workflow.add_node("GENERATE_LITERATURE", generate_with_formatted_sources)
else:
    workflow.add_node("GENERATE_LITERATURE", generate)

# Add TDS nodes
workflow.add_node("DECIDE_TDS_DATABASE", decide_tds_database)
workflow.add_node("ROUTE_TO_TDS_DATABASE", route_to_tds_database)
workflow.add_node("FILTER_TDS_DOCUMENTS", grade_documents)
if Config.USE_GENERATION_WITH_SOURCES_FOR_TDS:
    workflow.add_node("GENERATE_TDS", generate_with_formatted_sources)
else:
    workflow.add_node("GENERATE_TDS", generate)

# Add KNOWLEDGE BASE nodes
workflow.add_node("DECIDE_KNOWLEDGE_BASE_DATABASE", decide_knowledge_base_database)
workflow.add_node("ROUTE_TO_KNOWLEDGE_BASE_DATABASE", route_to_knowledge_base_database)
workflow.add_node("FILTER_KNOWLEDGE_BASE_DOCUMENTS", grade_documents)
if Config.USE_GENERATION_WITH_SOURCES_FOR_KNOWLEDGE_BASE:
    workflow.add_node("GENERATE_KNOWLEDGE_BASE", generate_with_formatted_sources)
else:
    workflow.add_node("GENERATE_KNOWLEDGE_BASE", generate)

# Join nodes to form graph edges
workflow.add_edge(START, "REFORMULATE_QUESTION")

# Add conditional edges for answer generation path routing
workflow.add_conditional_edges(
    "REFORMULATE_QUESTION",
    choose_answer_generation_path,
    {
        "generation_path:knowledge_base": "DECIDE_KNOWLEDGE_BASE_DATABASE",
        "generation_path:tds": "DECIDE_TDS_DATABASE",
        "generation_path:patent": "DECIDE_PATENT_DATABASE",
        "generation_path:literature": "DECIDE_LITERATURE_DATABASE",
        "generation_path:general": "GEN_INT_OUT_NODE",
    },
)



workflow.add_conditional_edges(
    "DECIDE_PATENT_DATABASE",
    decide_if_list_documents_query,
    {
        "list_documents_query_check_False_patents": "ROUTE_TO_PATENT_DATABASE",
        "list_documents_query_check_True_patents": "OBTAIN_LIST_OF_FILTERED_PATENTS",
    },
)

# LITERATURE - Add conditional routing for listing vs retrieval
workflow.add_conditional_edges(
    "DECIDE_LITERATURE_DATABASE",
    decide_if_list_documents_query,
    {
        "list_documents_query_check_False_literature": "ROUTE_TO_LITERATURE_DATABASE",
        "list_documents_query_check_True_literature": "OBTAIN_LIST_OF_FILTERED_LITERATURE",
    },
)

# TDS - Add conditional routing for listing vs retrieval
workflow.add_conditional_edges(
    "DECIDE_TDS_DATABASE",
    decide_if_list_documents_query,
    {
        "list_documents_query_check_False_tds": "ROUTE_TO_TDS_DATABASE",
        "list_documents_query_check_True_tds": "OBTAIN_LIST_OF_FILTERED_TDS",
    },
)
workflow.add_edge("ROUTE_TO_PATENT_DATABASE", "FILTER_PATENT_DOCUMENTS")
# workflow.add_edge("FILTER_PATENT_DOCUMENTS", "GENERATE_PATENT")
# workflow.add_edge("GENERATE_PATENT", END)
# Add conditional edges based on summarization need
workflow.add_conditional_edges(
    "FILTER_PATENT_DOCUMENTS",
    decide_if_summarization_needed,
    {
        "summarization_needed:True": "GENERATE_PATENT_SUMMARY",
        "summarization_needed:False": "GENERATE_PATENT",
    },
)
workflow.add_edge("OBTAIN_LIST_OF_FILTERED_PATENTS", END)
workflow.add_edge("OBTAIN_LIST_OF_FILTERED_TDS", END)
workflow.add_edge("OBTAIN_LIST_OF_FILTERED_LITERATURE", END)

workflow.add_edge("GENERATE_PATENT", END)

# Add literature workflow edges
workflow.add_edge("ROUTE_TO_LITERATURE_DATABASE", "FILTER_LITERATURE_DOCUMENTS")
workflow.add_edge("FILTER_LITERATURE_DOCUMENTS", "GENERATE_LITERATURE")
workflow.add_edge("GENERATE_LITERATURE", END)

# Add TDS workflow edges
workflow.add_edge("ROUTE_TO_TDS_DATABASE", "FILTER_TDS_DOCUMENTS")
workflow.add_edge("FILTER_TDS_DOCUMENTS", "GENERATE_TDS")
workflow.add_edge("GENERATE_TDS", END)

# Add KNOWLEDGE_BASE workflow edges
workflow.add_edge("DECIDE_KNOWLEDGE_BASE_DATABASE", "ROUTE_TO_KNOWLEDGE_BASE_DATABASE")
workflow.add_edge("ROUTE_TO_KNOWLEDGE_BASE_DATABASE", "FILTER_KNOWLEDGE_BASE_DOCUMENTS")
workflow.add_edge("FILTER_KNOWLEDGE_BASE_DOCUMENTS", "GENERATE_KNOWLEDGE_BASE")
workflow.add_edge("GENERATE_KNOWLEDGE_BASE", END)

# Add generic answer workflow edges
workflow.add_edge("GEN_INT_OUT_NODE", END)

# Compile the graph
app = workflow.compile()


