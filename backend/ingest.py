# This module will:
#   1) Loaded the medical textbook dataset from Hugging Face
#   2) Cleaned the text (removed tags, punctuation, whitespace)
#   3) Split the cleaned data into overlapping chunks (~1000 characters each)
#   4) Saved the result to data/chunks.json

import string
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import json
from pathlib import Path


def remove_tags(text: str) -> str:
    clean_text = re.sub(r'<.*?>', '', text)
    return clean_text


def clean_text(text: str, remove_punctuation: bool = True) -> str:
        # Remove HTML tags
        text = remove_tags(text)
        # Remove punctuation
        if remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text


def load_and_clean_data(dataset_name: str, split: str = 'train', text_key: str = 'text') -> list[str]:
    """
    Load and clean dataset from Hugging Face.

    :param dataset_name: Name of the dataset on HF
    :param split: Which data split to use ('train', 'test', etc.)
    :param text_key: Column in the dataset containing text to clean
    :return: List of cleaned text strings
    """
    dataset = load_dataset(dataset_name)[split]

    # Apply cleaning and make sure the output only includes the text column
    cleaned_dataset = dataset.map(
        lambda x: {text_key: clean_text(x[text_key])},
        remove_columns=dataset.column_names
    )

    return cleaned_dataset[text_key]


def split_into_chunks(texts: list[str], chunk_size: int = 1000, chunk_overlap: int = 100) -> list[str]:
    # This text splitter is the recommended one for generic text.
    # The default list is ["\n\n", "\n", " ", ""]. This has the effect of trying to keep all paragraphs
    # (and then sentences, and then words) together as long as possible,
    # as those would generically seem to be the strongest semantically related pieces of text.
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                              chunk_overlap=chunk_overlap)
    documents = splitter.create_documents(texts)
    return [doc.page_content for doc in documents]


def save_chunks(chunks: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as file:
        json.dump(chunks, file, indent=2)

    return None


def main():
    # Specify the name of the dataset
    dataset_name = "Gaoj124/medical_textbook_en"
    output_path = Path('../data/chunks.json')

    texts = load_and_clean_data(dataset_name)

    chunks = split_into_chunks(texts)

    save_chunks(chunks, output_path)


if __name__ == '__main__':
    main()
