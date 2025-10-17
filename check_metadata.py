from weaviate import Client
import json

client = Client("http://localhost:8080")

res = client.query.get(
    "ArtEmbedding",  # 👈 Correct class name
    ["filename", "style", "texture", "palette", "emotion"]
).with_limit(5).do()

print(json.dumps(res, indent=2))

