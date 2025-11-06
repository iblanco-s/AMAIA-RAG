# Ejercicio 05: Azure AI Search con Tags y Filtros

## 📚 Objetivo

Aprender a utilizar Azure AI Search para crear un sistema de búsqueda que permite filtrar resultados por categorías y tags. Este ejercicio demuestra cómo organizar y buscar contenido de manera selectiva usando metadatos.

## 🎯 Conceptos Clave

1. **Índices con campos filtrables**: Crear esquemas que incluyan campos de metadatos
2. **Tags y categorías**: Organizar documentos con múltiples dimensiones de clasificación
3. **Filtros OData**: Usar expresiones de filtrado para búsquedas precisas
4. **Búsqueda híbrida**: Combinar búsqueda semántica con filtros de metadatos

## 🏗️ Arquitectura

```
Documentos con metadatos
        ↓
    Indexación con tags
        ↓
  Azure AI Search Index
  (content + embeddings + tags)
        ↓
    Búsquedas filtradas
    - Por categoría
    - Por fuente
    - Por tags específicos
    - Combinaciones múltiples
```

## 📋 Prerequisitos

1. **Azure AI Search Service**
   - Crear un servicio en Azure Portal
   - Obtener endpoint y admin key

2. **Variables de entorno** (crear archivo `.env`):
   ```
   AZURE_SEARCH_ENDPOINT=https://your-service.search.windows.net
   AZURE_SEARCH_KEY=your-admin-key
   GITHUB_TOKEN=your-github-token
   ```

3. **Instalar dependencias**:
   ```bash
   pip install -r ../requirements.txt
   ```

## 🚀 Ejecución

```bash
python azure_search_with_tags.py
```

## 📊 Estructura del Índice

El ejercicio crea un índice con los siguientes campos:

| Campo | Tipo | Propósito |
|-------|------|-----------|
| `id` | String | Identificador único (key) |
| `content` | String | Contenido del documento (searchable) |
| `category` | String | Categoría principal (filterable) |
| `source` | String | Fuente del documento (filterable) |
| `tags` | Collection(String) | Lista de tags (filterable) |
| `contentVector` | Collection(Single) | Embedding para búsqueda semántica |

## 🔍 Ejemplos de Búsqueda

### 1. Filtrar por categoría
```python
search_by_tag(category="tecnologia")
```
Retorna todos los documentos clasificados como "tecnologia"

### 2. Filtrar por tags
```python
search_by_tag(tags=["ia", "python"])
```
Retorna documentos que contengan cualquiera de los tags especificados

### 3. Combinar filtros
```python
search_by_tag(category="tecnologia", source="manual_programacion")
```
Retorna documentos que cumplan TODAS las condiciones

### 4. Búsqueda híbrida
```python
hybrid_search("lenguajes de programación", category="tecnologia")
```
Busca semánticamente "lenguajes de programación" solo dentro de documentos de tecnología

## 💡 Casos de Uso

1. **Biblioteca digital**: Filtrar libros por género, autor, idioma
2. **Base de conocimiento empresarial**: Filtrar por departamento, tipo de documento, fecha
3. **E-commerce**: Filtrar productos por categoría, marca, características
4. **Sistema de tickets**: Filtrar por prioridad, estado, categoría del problema
5. **Archivo de noticias**: Filtrar por sección, fecha, autor, tags

## 🎓 Conceptos Avanzados

### Filtros OData
Azure AI Search usa sintaxis OData para filtros:

- **Igualdad**: `category eq 'tecnologia'`
- **Comparación**: `price gt 100`
- **Operadores lógicos**: `category eq 'tech' and price lt 50`
- **Funciones de colección**: `tags/any(t: t eq 'python')`

### Búsqueda Híbrida
Combina tres tipos de búsqueda:
1. **Texto completo**: Búsqueda por palabras clave
2. **Vectorial**: Búsqueda semántica por similitud
3. **Filtros**: Restricciones por metadatos

## 📝 Ejercicios Propuestos

1. **Agregar más campos**: Añade campos como `fecha`, `autor`, `idioma`
2. **Facetas**: Implementa faceted search para mostrar conteos por categoría
3. **Scoring profiles**: Crea perfiles que den más peso a ciertos campos
4. **Sugerencias**: Implementa autocompletado basado en tags
5. **Paginación**: Añade soporte para paginar resultados

## 🔗 Referencias

- [Azure AI Search Documentation](https://learn.microsoft.com/en-us/azure/search/)
- [OData Filter Syntax](https://learn.microsoft.com/en-us/azure/search/search-query-odata-filter)
- [Vector Search](https://learn.microsoft.com/en-us/azure/search/vector-search-overview)
- [Hybrid Search](https://learn.microsoft.com/en-us/azure/search/hybrid-search-overview)
