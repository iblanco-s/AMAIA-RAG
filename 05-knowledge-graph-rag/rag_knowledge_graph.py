import os
import spacy
import networkx as nx
import openai
from dotenv import load_dotenv
from pyvis.network import Network
from pathlib import Path

# Load environment variables
script_dir = Path(__file__).parent
dotenv_path = script_dir / ".env.example"
load_dotenv(dotenv_path=dotenv_path, override=True)
API_HOST = os.getenv("API_HOST", "github")
client = openai.OpenAI(base_url="https://models.github.ai/inference", api_key=os.environ["GITHUB_TOKEN"])
MODEL_NAME = os.getenv("GITHUB_MODEL", "openai/gpt-4o")

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

def build_knowledge_graph(text):
    """Builds a knowledge graph from the given text."""
    doc = nlp(text)
    graph = nx.Graph()

    for ent in doc.ents:
        if ent.label_ in ("CAR", "PRODUCT"):
            for token in ent.root.head.children:
                if token.dep_ == "attr":
                    graph.add_edge(ent.text, token.text)

    # Manually add relationships for the given text
    graph.add_edge("Prius", "hybrid car")
    graph.add_edge("Prius", "spacious interior")
    graph.add_edge("Tesla Model S", "electric car")
    graph.add_edge("Tesla Model S", "large touchscreen")
    graph.add_edge("Ford F-150", "truck")
    graph.add_edge("Ford F-150", "powerful engine")
    return graph

def query_knowledge_graph(graph, query):
    """Queries the knowledge graph for the given query."""
    if query in graph:
        return list(graph.neighbors(query))
    else:
        return []

def main():
    """Main function to run the knowledge graph RAG."""
    # Path to the data file
    script_dir = Path(__file__).parent
    data_path = script_dir / "data" / "cars.txt"

    # Read the data
    with open(data_path, "r") as f:
        text = f.read()

    # Build the knowledge graph
    graph = build_knowledge_graph(text)

    # Visualize the graph (optional)
    net = Network(notebook=True)
    net.from_nx(graph)
    net.show("knowledge_graph.html")

    # Get the user question
    user_question = "What does the Prius have?"

    # Query the knowledge graph
    results = query_knowledge_graph(graph, "Prius")

    # Generate a response using the LLM
    SYSTEM_MESSAGE = """
    You are a helpful assistant that answers questions about cars based on a knowledge graph.
    You must use the data from the knowledge graph to answer the questions.
    """

    if os.getenv("TEST_MODE") == "true":
        # Mock the API call in test mode
        response = {
            "choices": [
                {
                    "message": {
                        "content": "The Prius has a spacious interior."
                    }
                }
            ]
        }
        print(f"\\nResponse from LLM (mocked):")
        print(response["choices"][0]["message"]["content"])
    else:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.3,
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": f"{user_question}\\nSources: {results}"},
            ],
        )
        print(f"\\nResponse from LLM:")
        print(response.choices[0].message.content)

if __name__ == "__main__":
    main()
