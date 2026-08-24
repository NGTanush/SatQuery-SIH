import os
import sys
import logging
from PIL import Image

# Add workspace directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("satquery.datasets.real")

def fetch_samples():
    try:
        from datasets import load_dataset
        logger.info("Loading dmarsili/RSVQA-LR-2k dataset from HuggingFace...")
        # Load low-resolution RSVQA dataset (small split or stream if possible)
        dataset = load_dataset("dmarsili/RSVQA-LR-2k", split="validation", streaming=True)
        
        # Get first 3 samples
        iterator = iter(dataset)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        rsvqa_dir = os.path.join(base_dir, "rsvqa")
        os.makedirs(rsvqa_dir, exist_ok=True)
        
        logger.info("Extracting sample images and QA pairs...")
        samples_metadata = []
        
        for idx in range(3):
            try:
                sample = next(iterator)
                # Structure of RSVQA-LR-2k: has 'image', 'question', 'answer', 'category'
                # Let's inspect keys
                logger.info(f"Sample keys: {list(sample.keys())}")
                
                img = sample.get("image")
                question = sample.get("question", "What is visible in the image?")
                answer = sample.get("answer", "")
                
                img_name = f"rsvqa_sample_{idx}.png"
                img_path = os.path.join(rsvqa_dir, img_name)
                
                # Save PIL image
                if isinstance(img, Image.Image):
                    img.save(img_path)
                else:
                    # If it's a dict/path/etc.
                    logger.warning(f"Image type: {type(img)}")
                    
                samples_metadata.append({
                    "id": f"rsvqa_{idx}",
                    "image_name": img_name,
                    "question": question,
                    "answer": answer
                })
                logger.info(f"Saved real RSVQA image to {img_path}")
                logger.info(f"  - Q: {question}")
                logger.info(f"  - A: {answer}")
            except StopIteration:
                break
                
        # Write metadata file
        import json
        with open(os.path.join(rsvqa_dir, "metadata.json"), "w") as f:
            json.dump(samples_metadata, f, indent=2)
        logger.info("Successfully fetched real RSVQA low-res samples.")
        
    except Exception as e:
        logger.error(f"Failed to fetch real RSVQA data: {str(e)}")
        logger.info("Fallback: Proceeding with local offline generated dataset.")

if __name__ == "__main__":
    fetch_samples()
