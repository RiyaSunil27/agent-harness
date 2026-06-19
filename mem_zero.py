import os
from mem0 import MemoryClient
 
client = MemoryClient(api_key=os.getenv("MEM0_API_KEY", "your-api-key"))
 
# Add a memory
messages = [
    {"role": "user", "content": "I'm a non vegetarian but i dont eat eggs and i love olives."},
    {"role": "assistant", "content": "Got it! I'll remember your dietary preferences."},
]
client.add(messages, user_id="riya")
 
# Search memories
results = client.search(
    "What are my dietary preferences?",
    filters={'user_id':"riya"}
)
print(results)