"""
Ejercicio: Azure AI Search con Tags para Filtrado de Búsquedas

Este ejercicio demuestra cómo:
1. Crear un índice en Azure AI Search con campos de metadatos (tags)
2. Indexar documentos con diferentes categorías/tags
3. Realizar búsquedas filtradas por tags
4. Combinar búsqueda de texto con filtros de tags

Requisitos:
- Azure AI Search service creado
- Variables de entorno configuradas en .env:
  - AZURE_SEARCH_ENDPOINT
  - AZURE_SEARCH_KEY
  - GITHUB_TOKEN (para embeddings)
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
INDEX_NAME = "documents-with-tags"

# Cliente de OpenAI para generar embeddings
openai_client = openai.OpenAI(
    base_url="https://models.github.ai/inference",
    api_key=os.environ["GITHUB_TOKEN"]
)

# Clientes de Azure AI Search
credential = AzureKeyCredential(AZURE_SEARCH_KEY)
index_client = SearchIndexClient(endpoint=AZURE_SEARCH_ENDPOINT, credential=credential)
search_client = SearchClient(endpoint=AZURE_SEARCH_ENDPOINT, index_name=INDEX_NAME, credential=credential)


def create_search_index():
    """
    Crea un índice en Azure AI Search con campos para:
    - id: Identificador único
    - content: Contenido del documento (searchable)
    - category: Tag/categoría del documento (filterable)
    - source: Fuente del documento (filterable)
    - tags: Lista de tags (filterable, collection)
    - contentVector: Vector de embeddings para búsqueda semántica
    """
    print(f"\n🔧 Creando índice '{INDEX_NAME}'...")

    # Definir los campos del índice
    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            sortable=True,
            filterable=True,
        ),
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
            searchable=True,
        ),
        SimpleField(
            name="category",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SimpleField(
            name="source",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
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
            vector_search_dimensions=1536,  # Dimensiones de text-embedding-3-small
            vector_search_profile_name="myHnswProfile",
        ),
    ]

    # Configurar búsqueda vectorial
    vector_search = VectorSearch(
        profiles=[
            VectorSearchProfile(
                name="myHnswProfile",
                algorithm_configuration_name="myHnsw",
            )
        ],
        algorithms=[
            HnswAlgorithmConfiguration(
                name="myHnsw"
            )
        ],
    )

    # Crear el índice
    index = SearchIndex(
        name=INDEX_NAME,
        fields=fields,
        vector_search=vector_search,
    )

    # Eliminar índice si ya existe
    try:
        index_client.delete_index(INDEX_NAME)
        print(f"  ✓ Índice anterior eliminado")
    except Exception:
        pass

    # Crear nuevo índice
    index_client.create_index(index)
    print(f"  ✓ Índice '{INDEX_NAME}' creado exitosamente")
    print(f"  ✓ Campos configurados: id, content, category, source, tags, contentVector")


def generate_embedding(text: str) -> List[float]:
    """Genera embedding para un texto usando OpenAI"""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def index_documents():
    """
    Indexa documentos de ejemplo con diferentes categorías y tags
    """
    print(f"\n📝 Indexando documentos con tags...")

    # Documentos de ejemplo con diferentes categorías y tags
    documents = [
        {
            "id": "doc1",
            "content": "Python es un lenguaje de programación interpretado de alto nivel. Es ideal para desarrollo web, ciencia de datos y automatización.",
            "category": "tecnologia",
            "source": "manual_programacion",
            "tags": ["python", "programacion", "desarrollo"],
        },
        {
            "id": "doc2",
            "content": "JavaScript es el lenguaje de programación más usado en desarrollo web frontend. También se usa en backend con Node.js.",
            "category": "tecnologia",
            "source": "manual_programacion",
            "tags": ["javascript", "web", "frontend"],
        },
        {
            "id": "doc3",
            "content": "Azure AI Search es un servicio de búsqueda en la nube que proporciona APIs REST y SDK para indexar y consultar datos.",
            "category": "cloud",
            "source": "documentacion_azure",
            "tags": ["azure", "search", "cloud"],
        },
        {
            "id": "doc4",
            "content": "La diabetes tipo 2 es una enfermedad crónica que afecta la forma en que el cuerpo procesa el azúcar en la sangre.",
            "category": "salud",
            "source": "enciclopedia_medica",
            "tags": ["diabetes", "salud", "enfermedad"],
        },
        {
            "id": "doc5",
            "content": "El ejercicio cardiovascular ayuda a fortalecer el corazón y mejorar la circulación sanguínea.",
            "category": "salud",
            "source": "guia_ejercicios",
            "tags": ["ejercicio", "salud", "cardio"],
        },
        {
            "id": "doc6",
            "content": "RAG (Retrieval Augmented Generation) combina búsqueda de documentos con modelos de lenguaje para generar respuestas contextualmente relevantes.",
            "category": "ai",
            "source": "documentacion_ia",
            "tags": ["rag", "ia", "llm"],
        },
        {
            "id": "doc7",
            "content": "Los embeddings son representaciones vectoriales de texto que capturan el significado semántico de las palabras y frases.",
            "category": "ai",
            "source": "documentacion_ia",
            "tags": ["embeddings", "ia", "nlp"],
        },
        {
            "id": "doc8",
            "content": "Docker es una plataforma para desarrollar, enviar y ejecutar aplicaciones en contenedores.",
            "category": "tecnologia",
            "source": "manual_devops",
            "tags": ["docker", "contenedores", "devops"],
        },
    ]

    # Generar embeddings y preparar documentos para indexar
    docs_to_index = []
    for doc in documents:
        print(f"  Procesando: {doc['id']} - Categoría: {doc['category']}")

        # Generar embedding del contenido
        embedding = generate_embedding(doc["content"])

        # Preparar documento con embedding
        doc_with_embedding = {
            "id": doc["id"],
            "content": doc["content"],
            "category": doc["category"],
            "source": doc["source"],
            "tags": doc["tags"],
            "contentVector": embedding,
        }
        docs_to_index.append(doc_with_embedding)

    # Subir documentos al índice
    result = search_client.upload_documents(documents=docs_to_index)
    print(f"  ✓ {len(docs_to_index)} documentos indexados exitosamente")

    return documents


def search_by_tag(category: str = None, source: str = None, tags: List[str] = None):
    """
    Búsqueda filtrada por tags usando filtros OData

    Args:
        category: Filtrar por categoría
        source: Filtrar por fuente
        tags: Lista de tags para filtrar
    """
    filters = []

    if category:
        filters.append(f"category eq '{category}'")

    if source:
        filters.append(f"source eq '{source}'")

    if tags:
        tag_filters = [f"tags/any(t: t eq '{tag}')" for tag in tags]
        filters.append(f"({' or '.join(tag_filters)})")

    filter_expression = " and ".join(filters) if filters else None

    print(f"\n🔍 Búsqueda con filtros:")
    if category:
        print(f"  - Categoría: {category}")
    if source:
        print(f"  - Fuente: {source}")
    if tags:
        print(f"  - Tags: {tags}")

    if filter_expression:
        print(f"  Expresión de filtro: {filter_expression}")

    # Realizar búsqueda
    results = search_client.search(
        search_text="*",  # Buscar todos los documentos
        filter=filter_expression,
        select=["id", "content", "category", "source", "tags"],
        top=10
    )

    print(f"\n📋 Resultados:")
    count = 0
    for result in results:
        count += 1
        print(f"\n  {count}. ID: {result['id']}")
        print(f"     Categoría: {result['category']}")
        print(f"     Fuente: {result['source']}")
        print(f"     Tags: {', '.join(result['tags'])}")
        print(f"     Contenido: {result['content'][:100]}...")

    if count == 0:
        print("  ⚠️  No se encontraron resultados")

    return count


def hybrid_search(query: str, category: str = None, tags: List[str] = None):
    """
    Búsqueda híbrida: combina búsqueda semántica con filtros de tags

    Args:
        query: Texto de búsqueda
        category: Filtrar por categoría
        tags: Lista de tags para filtrar
    """
    # Generar embedding de la consulta
    query_vector = generate_embedding(query)

    # Construir filtros
    filters = []
    if category:
        filters.append(f"category eq '{category}'")
    if tags:
        tag_filters = [f"tags/any(t: t eq '{tag}')" for tag in tags]
        filters.append(f"({' or '.join(tag_filters)})")

    filter_expression = " and ".join(filters) if filters else None

    print(f"\n🔍 Búsqueda híbrida:")
    print(f"  - Query: '{query}'")
    if category:
        print(f"  - Categoría: {category}")
    if tags:
        print(f"  - Tags: {tags}")

    # Realizar búsqueda híbrida (texto + vectorial + filtros)
    results = search_client.search(
        search_text=query,
        vector_queries=[{
            "vector": query_vector,
            "k_nearest_neighbors": 5,
            "fields": "contentVector"
        }],
        filter=filter_expression,
        select=["id", "content", "category", "source", "tags"],
        top=5
    )

    print(f"\n📋 Resultados (ordenados por relevancia):")
    count = 0
    for result in results:
        count += 1
        score = result.get('@search.score', 0)
        print(f"\n  {count}. ID: {result['id']} (Score: {score:.4f})")
        print(f"     Categoría: {result['category']}")
        print(f"     Tags: {', '.join(result['tags'])}")
        print(f"     Contenido: {result['content'][:150]}...")

    if count == 0:
        print("  ⚠️  No se encontraron resultados")

    return count


def main():
    """Función principal que ejecuta todas las demostraciones"""

    print("=" * 70)
    print("  Azure AI Search - Ejercicio con Tags y Filtros")
    print("=" * 70)

    # 1. Crear índice
    create_search_index()

    # 2. Indexar documentos con tags
    documents = index_documents()

    # 3. Ejemplo 1: Buscar todos los documentos de tecnología
    search_by_tag(category="tecnologia")

    # 4. Ejemplo 2: Buscar documentos de salud
    search_by_tag(category="salud")

    # 5. Ejemplo 3: Buscar documentos con tag específico
    search_by_tag(tags=["ia"])

    # 6. Ejemplo 4: Combinar categoría y fuente
    search_by_tag(category="tecnologia", source="manual_programacion")

    # 7. Ejemplo 5: Búsqueda híbrida - buscar sobre programación solo en tecnología
    hybrid_search("lenguajes de programación", category="tecnologia")

    # 8. Ejemplo 6: Búsqueda híbrida - buscar sobre salud
    hybrid_search("cómo mejorar la salud del corazón", category="salud")

    # 9. Ejemplo 7: Búsqueda híbrida - buscar IA sin filtros
    hybrid_search("inteligencia artificial y modelos de lenguaje")

    # 10. Ejemplo 8: Búsqueda híbrida - buscar IA solo en documentos con tag 'rag'
    hybrid_search("búsqueda de información", tags=["rag"])

    print("\n" + "=" * 70)
    print("  ✅ Ejercicio completado!")
    print("=" * 70)
    print("\n💡 Puntos clave aprendidos:")
    print("  1. Creación de índices con campos filtrables (category, source, tags)")
    print("  2. Indexación de documentos con metadatos y tags")
    print("  3. Búsquedas filtradas usando expresiones OData")
    print("  4. Búsquedas híbridas combinando texto, vectores y filtros")
    print("  5. Uso de tags para organizar y filtrar contenido por categorías")


if __name__ == "__main__":
    main()
