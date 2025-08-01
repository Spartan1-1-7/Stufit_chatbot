from qdrant_client import QdrantClient

client = QdrantClient(path="qdrant_db")
collection_name = "your_collection"

scroll_result = client.scroll(collection_name=collection_name, limit=5, with_vectors=True)
points = scroll_result[0]

if len(points) == 0:
    print("No points found: Your Qdrant store may be empty or misconfigured.")
else:
    for point in points:
        print(f"Point ID: {point.id}")
        if hasattr(point, "vector") and point.vector is not None:
            print(f"First 5 vector values: {point.vector[:5]}...")
        else:
            print("No vector stored for this point.")
        print(f"Payload chunk preview: {point.payload.get('chunk_text', '(none)') if point.payload else '(none)'}")
        print("="*40)
