"""
Ejercicio avanzado: Cargar documentos desde archivo JSON con metadatos

Este script demuestra cómo:
1. Cargar documentos desde un archivo JSON con metadatos
2. Procesar y generar embeddings para cada documento
3. Indexar todos los documentos en Azure AI Search
4. Realizar búsquedas interactivas con filtros

Uso:
    python azure_search_from_file.py
"""

import json
import os
import pathlib
from typing import List, Dict, Any

import openai
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
)
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv(override=True)

# Configuración
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
INDEX_NAME = "documents-from-file"
DATA_FILE = "data/documents_with_metadata.json"

# Cliente de OpenAI para generar embeddings
openai_client = openai.OpenAI(
    base_url="https://models.github.ai/inference",
    api_key=os.environ["GITHUB_TOKEN"]
)

# Clientes de Azure AI Search
credential = AzureKeyCredential(AZURE_SEARCH_KEY)
index_client = SearchIndexClient(endpoint=AZURE_SEARCH_ENDPOINT, credential=credential)


def create_search_index():
    """Crea el índice en Azure AI Search"""
    print(f"\n🔧 Creando índice '{INDEX_NAME}'...")

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
        SearchableField(name="content", type=SearchFieldDataType.String, searchable=True),
        SimpleField(name="category", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="source", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(
            name="tags",
            type=SearchFieldDataType.Collection(SearchFieldDataType.String),
            filterable=True,
            facetable=True,
        ),
        SearchField(
            name="contentVector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="myHnswProfile",
        ),
    ]

    vector_search = VectorSearch(
        profiles=[VectorSearchProfile(name="myHnswProfile", algorithm_configuration_name="myHnsw")],
        algorithms=[HnswAlgorithmConfiguration(name="myHnsw")],
    )

    index = SearchIndex(name=INDEX_NAME, fields=fields, vector_search=vector_search)

    try:
        index_client.delete_index(INDEX_NAME)
        print(f"  ✓ Índice anterior eliminado")
    except Exception:
        pass

    index_client.create_index(index)
    print(f"  ✓ Índice '{INDEX_NAME}' creado exitosamente")


def generate_embedding(text: str) -> List[float]:
    """Genera embedding para un texto"""
    response = openai_client.embeddings.create(model="text-embedding-3-small", input=text)
    return response.data[0].embedding


def load_and_index_documents(file_path: str):
    """
    Carga documentos desde un archivo JSON y los indexa en Azure AI Search

    Args:
        file_path: Ruta al archivo JSON con los documentos
    """
    print(f"\n📁 Cargando documentos desde '{file_path}'...")

    # Leer archivo JSON
    data_path = pathlib.Path(os.path.dirname(__file__)) / file_path
    with open(data_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)

    print(f"  ✓ {len(documents)} documentos cargados")

    # Analizar estadísticas
    categories = {}
    sources = {}
    all_tags = set()

    for doc in documents:
        # Contar categorías
        cat = doc.get('category', 'sin_categoria')
        categories[cat] = categories.get(cat, 0) + 1

        # Contar fuentes
        src = doc.get('source', 'sin_fuente')
        sources[src] = sources.get(src, 0) + 1

        # Recopilar tags
        all_tags.update(doc.get('tags', []))

    print(f"\n📊 Estadísticas:")
    print(f"  Categorías: {len(categories)}")
    for cat, count in sorted(categories.items()):
        print(f"    - {cat}: {count} documentos")
    print(f"  Fuentes: {len(sources)}")
    for src, count in sorted(sources.items()):
        print(f"    - {src}: {count} documentos")
    print(f"  Tags únicos: {len(all_tags)}")

    print(f"\n⚙️  Generando embeddings e indexando...")

    search_client = SearchClient(endpoint=AZURE_SEARCH_ENDPOINT, index_name=INDEX_NAME, credential=credential)

    docs_to_index = []
    for i, doc in enumerate(documents, 1):
        print(f"  [{i}/{len(documents)}] Procesando: {doc['id']}", end='\r')

        # Generar embedding
        embedding = generate_embedding(doc['content'])

        # Preparar documento
        doc_with_embedding = {
            "id": doc['id'],
            "content": doc['content'],
            "category": doc.get('category', 'sin_categoria'),
            "source": doc.get('source', 'sin_fuente'),
            "tags": doc.get('tags', []),
            "contentVector": embedding,
        }
        docs_to_index.append(doc_with_embedding)

    # Indexar todos los documentos
    result = search_client.upload_documents(documents=docs_to_index)
    print(f"\n  ✓ {len(docs_to_index)} documentos indexados exitosamente")

    return documents


def interactive_search():
    """
    Modo interactivo para realizar búsquedas con filtros
    """
    search_client = SearchClient(endpoint=AZURE_SEARCH_ENDPOINT, index_name=INDEX_NAME, credential=credential)

    print("\n" + "=" * 70)
    print("  🔍 Modo de búsqueda interactiva")
    print("=" * 70)
    print("\nEjemplos de búsquedas:")
    print("  1. Buscar por categoría: category:tecnologia")
    print("  2. Buscar por tag: tag:python")
    print("  3. Buscar por fuente: source:manual_programacion")
    print("  4. Búsqueda de texto: docker contenedores")
    print("  5. Combinar: category:ai inteligencia artificial")
    print("  6. Salir: exit o quit\n")

    while True:
        query = input("🔍 Búsqueda: ").strip()

        if query.lower() in ['exit', 'quit', 'salir']:
            print("¡Hasta luego!")
            break

        if not query:
            continue

        # Parsear query para extraer filtros
        search_text = []
        filters = []

        for term in query.split():
            if ':' in term:
                key, value = term.split(':', 1)
                if key == 'category':
                    filters.append(f"category eq '{value}'")
                elif key == 'source':
                    filters.append(f"source eq '{value}'")
                elif key == 'tag':
                    filters.append(f"tags/any(t: t eq '{value}')")
            else:
                search_text.append(term)

        search_query = ' '.join(search_text) if search_text else '*'
        filter_expression = ' and '.join(filters) if filters else None

        # Generar embedding si hay texto de búsqueda
        vector_queries = None
        if search_text:
            query_vector = generate_embedding(search_query)
            vector_queries = [{
                "vector": query_vector,
                "k_nearest_neighbors": 5,
                "fields": "contentVector"
            }]

        print(f"\n📋 Buscando...")
        if filter_expression:
            print(f"   Filtros: {filter_expression}")

        # Realizar búsqueda
        results = search_client.search(
            search_text=search_query,
            vector_queries=vector_queries,
            filter=filter_expression,
            select=["id", "content", "category", "source", "tags"],
            top=5
        )

        # Mostrar resultados
        count = 0
        for result in results:
            count += 1
            score = result.get('@search.score', 0)
            print(f"\n  {count}. {result['id']} (Score: {score:.4f})")
            print(f"     📂 Categoría: {result['category']}")
            print(f"     📄 Fuente: {result['source']}")
            print(f"     🏷️  Tags: {', '.join(result['tags'])}")
            print(f"     📝 {result['content'][:200]}...")

        if count == 0:
            print("  ⚠️  No se encontraron resultados")

        print()


def demo_searches():
    """Ejecuta algunas búsquedas de demostración"""
    search_client = SearchClient(endpoint=AZURE_SEARCH_ENDPOINT, index_name=INDEX_NAME, credential=credential)

    demos = [
        {
            "title": "Todos los documentos de IA",
            "filter": "category eq 'ai'",
            "search_text": "*"
        },
        {
            "title": "Documentos con tag 'python'",
            "filter": "tags/any(t: t eq 'python')",
            "search_text": "*"
        },
        {
            "title": "Búsqueda: 'modelos de lenguaje' en categoría IA",
            "filter": "category eq 'ai'",
            "search_text": "modelos de lenguaje",
            "use_vector": True
        },
        {
            "title": "Documentos de salud sobre corazón",
            "filter": "category eq 'salud'",
            "search_text": "corazón cardiovascular",
            "use_vector": True
        },
    ]

    print("\n" + "=" * 70)
    print("  🎬 Demostraciones de búsqueda")
    print("=" * 70)

    for demo in demos:
        print(f"\n🔍 {demo['title']}")
        print(f"   Query: '{demo['search_text']}'")
        print(f"   Filtro: {demo['filter']}")

        vector_queries = None
        if demo.get('use_vector') and demo['search_text'] != '*':
            query_vector = generate_embedding(demo['search_text'])
            vector_queries = [{
                "vector": query_vector,
                "k_nearest_neighbors": 3,
                "fields": "contentVector"
            }]

        results = search_client.search(
            search_text=demo['search_text'],
            vector_queries=vector_queries,
            filter=demo['filter'],
            select=["id", "category", "tags"],
            top=3
        )

        count = 0
        for result in results:
            count += 1
            print(f"   {count}. {result['id']} - {result['category']} - Tags: {', '.join(result['tags'])}")

        if count == 0:
            print("   ⚠️  Sin resultados")


def main():
    """Función principal"""
    print("=" * 70)
    print("  Azure AI Search - Indexación desde archivo JSON")
    print("=" * 70)

    # 1. Crear índice
    create_search_index()

    # 2. Cargar e indexar documentos
    load_and_index_documents(DATA_FILE)

    # 3. Ejecutar búsquedas de demostración
    demo_searches()

    # 4. Modo interactivo
    print("\n" + "=" * 70)
    interactive_search()


if __name__ == "__main__":
    main()
