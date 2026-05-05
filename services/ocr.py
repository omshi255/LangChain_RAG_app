import pytesseract
from PIL import Image
import logging

logger = logging.getLogger(__name__)

# Update path if needed
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def extract_text_from_image(image_path: str) -> str:
    try:
        image = Image.open(image_path).convert("RGB")
        text = pytesseract.image_to_string(image)

        cleaned = text.strip()
        logger.info(f"OCR extracted: {cleaned[:100]}...")

        return cleaned

    except Exception as e:
        logger.error(f"OCR failed: {e}")
        raise