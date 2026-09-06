import json
import os

MEMORY_FILE = "memory.json"

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return {}
    try:
        with open(MEMORY_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading memory: {e}")
        return {}

def save_memory(data):
    try:
        with open(MEMORY_FILE, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Error saving memory: {e}")

def get_memory(key):
    memory = load_memory()
    return memory.get(key.lower())

def update_memory(key, value):
    memory = load_memory()
    memory[key.lower()] = value
    save_memory(memory)

def parse_and_remember(text):
    text = text.lower().strip()
    
    # Name patterns
    if "my name is" in text:
        name = text.split("my name is")[-1].strip().title()
        update_memory("user_name", name)
        return f"I will remember that your name is {name}."
    
    if "call me" in text:
        name = text.split("call me")[-1].strip().title()
        update_memory("user_name", name)
        return f"Sure, I will call you {name} from now on."
        
    # Generic memory: "remember that my favorite color is blue"
    if text.startswith("remember that"):
        content = text.replace("remember that", "", 1).strip()
        if "is" in content:
            key, val = content.split("is", 1)
            update_memory(key.strip(), val.strip())
            return f"Got it. I've remembered that {key.strip()} is {val.strip()}."

    return None
