from openai import OpenAI
from pdf2image import convert_from_path
import base64
from io import BytesIO

client = OpenAI(
    base_url="http://0.0.0.0:8000/v1",
    api_key="dummy"
)

# Convert PDF to images (one per page)
images = convert_from_path("peek.pdf", dpi=300)

results = []
for page_num, image in enumerate(images, 1):
    # Convert PIL image to base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    response = client.chat.completions.create(
        model="datalab-to/chandra",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}},
                    {"type": "text", "text": "Convert this page to markdown preserving layout"}
                ]
            }
        ],
        max_tokens=4096
    )
    
    results.append({
        "page": page_num,
        "content": response.choices[0].message.content
    })

# Combine all pages
full_document = "\n\n---\n\n".join([r["content"] for r in results])
