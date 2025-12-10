import requests
import json
import os
import base64


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable is not set")

model_input_dir = (
    "/home/ben/devel/iai/src/cognitive_robot_abstract_machine/scripts/model_input"
)
image_path = os.path.join(model_input_dir, "original_render_1.png")
highlighted_image_path = os.path.join(model_input_dir, "group_3_render_1.png")

# Encode the image
base64_original_scene = encode_image(image_path)
base64_highlighted_scene = encode_image(highlighted_image_path)

object_taxonomy = open(os.path.join(model_input_dir, "object_taxonomy.txt"), "r").read()
spatial_relations = open(
    os.path.join(model_input_dir, "spatial_relations.txt"), "r"
).read()

response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "ai.uni-bremen.de",  # Optional. Site URL for rankings on openrouter.ai.
        "X-Title": "Uni Bremen",  # Optional. Site title for rankings on openrouter.ai.
    },
    data=json.dumps(
        {
            "model": "qwen/qwen2.5-vl-72b-instruct",
            "messages": [
                {
                    "role": "system",
                    "content": """You are a semantic perception system for robotic scene understanding.

## Your Task
Analyze images and classify objects according to a given ontology. You will receive:
1. An image of a scene with original textures
2. An image with specific objects highlighted in distinct colors

Focus ONLY on the highlighted objects. For each:
- Identify its class from the provided taxonomy
- If no suitable class exists, propose a new subclass under the most appropriate parent
- Identify spatial relations between the highlighted objects

## Output Schema
Respond with valid JSON:
{
  "objects": [
    {
      "highlight_color": "string (the color used to highlight this object: red, green, blue, etc.)",
      "classification": {
        "class": "string (class name from taxonomy, or your proposed new class)",
        "superclass": "string (parent class in taxonomy)",
        "is_new_class": "boolean (true if you're proposing a new class)",
        "new_class_justification": "string | null (if is_new_class, explain why existing classes don't fit)"
      },
      "confidence": "number (0-1)"
    }
  ],
  "spatial_relations": [
    {
      "subject": "string (highlight_color of subject object)",
      "relation": "string (relation from the provided vocabulary)",
      "object": "string (highlight_color of object)"
    }
  ],
  "notes": "string | null (any ambiguities or uncertainties)"
}""",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""## Object Taxonomy (Hierarchy)
{object_taxonomy}

## Spatial Relations Vocabulary
{spatial_relations}

## Images

Image 1: Original scene with natural textures
Image 2: Same scene with target objects highlighted in distinct colors

Identify the highlighted objects and their spatial relations.
""",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_original_scene}"
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_highlighted_scene}"
                            },
                        },
                    ],
                },
            ],
        }
    ),
)

print(response.json())
