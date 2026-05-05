import os
import uuid
import base64
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image

from services.ocr import extract_text_from_image
from services.embeddings import embed_texts
from services.vector_store import get_index
from groq import Groq
from config import get_settings

router = APIRouter()
settings = get_settings()

UPLOAD_DIR = "temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def encode_image_to_base64(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def convert_to_supported_format(file_path: str) -> str:
    """Convert any image format to JPEG for compatibility with Groq."""
    ext = file_path.lower().split(".")[-1]
    supported = ["jpg", "jpeg", "png", "gif", "webp"]

    if ext not in supported:
        converted_path = file_path.rsplit(".", 1)[0] + "_converted.jpg"
        img = Image.open(file_path).convert("RGB")
        img.save(converted_path, "JPEG", quality=95)
        return converted_path

    return file_path  # already supported, no conversion needed


def describe_image_with_vision(file_path: str) -> str:
    """Use Groq vision model to describe any image."""
    client = Groq(api_key=settings.groq_api_key)

    # Convert to supported format first
    converted_path = convert_to_supported_format(file_path)
    base64_image = encode_image_to_base64(converted_path)

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Please analyze this image completely and describe:\n"
                            "1. What is shown in the image\n"
                            "2. Any text visible\n"
                            "3. Colors, objects, people, animals if any\n"
                            "4. Any other important details\n"
                            "Be as detailed as possible."
                        ),
                    },
                ],
            }
        ],
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


@router.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Save file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        extracted_text = None
        source_type = "ocr"

        # Step 1: Try OCR first
        try:
            ocr_text = extract_text_from_image(file_path)
            if ocr_text and len(ocr_text.strip()) > 10:
                extracted_text = ocr_text
                source_type = "ocr"
        except Exception:
            pass  # OCR failed, will fallback to vision

        # Step 2: If OCR gave nothing, use vision model
        if not extracted_text:
            try:
                extracted_text = describe_image_with_vision(file_path)
                source_type = "vision"
            except Exception as vision_err:
                raise HTTPException(
                    status_code=500,
                    detail=f"Both OCR and Vision failed: {str(vision_err)}"
                )

        # Step 3: Final check
        if not extracted_text:
            raise HTTPException(
                status_code=400,
                detail="Could not extract any information from the image."
            )

        # Chunk
        texts = [extracted_text]

        # Generate document_id
        document_id = str(uuid.uuid4())

        # Embed
        embeddings = embed_texts(texts)

        # Store in Pinecone
        index = get_index()

        vectors = []
        for i, (text, emb) in enumerate(zip(texts, embeddings)):
            vectors.append({
                "id": f"{document_id}_{i}",
                "values": emb,
                "metadata": {
                    "text": text,
                    "document_id": document_id,
                    "source": source_type,
                }
            })

        index.upsert(vectors=vectors)

        return {
            "message": "Image processed and stored successfully",
            "document_id": document_id,
            "source_type": source_type,
            "extracted_text": extracted_text[:300]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Image processing failed: {str(e)}"
        )