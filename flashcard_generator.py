# flashcard_generator.py

import spacy
import random
from nltk.corpus import wordnet
import nltk
import os # Import os for checking file existence

# --- NLTK Data Check (Important for script execution) ---
# It's better to check if the data is already present on the disk
# rather than relying solely on exceptions which can be broad.

# Define the NLTK data paths - typically found in ~/nltk_data or C:\nltk_data
nltk_data_path = os.path.expanduser('~/nltk_data') # Common path on Linux/macOS
if os.name == 'nt': # If Windows, prioritize APPDATA
    nltk_data_path = os.path.join(os.getenv('APPDATA'), 'nltk_data')

required_nltk_corpora = {
    'wordnet': 'corpora/wordnet',
    'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
    'punkt': 'tokenizers/punkt',
    'omw-1.4': 'corpora/omw-1.4'
}

for corp_name, corp_path in required_nltk_corpora.items():
    # Construct potential full path to check for existence
    full_corp_check_path = os.path.join(nltk_data_path, corp_path)
    # NLTK stores data in subdirectories; checking for its existence is usually sufficient
    # For example, 'wordnet' exists if '.../nltk_data/corpora/wordnet' directory is there
    # For 'punkt' and 'averaged_perceptron_tagger', it's a bit different,
    # but nltk.data.find is the most reliable way to check for NLTK's internal knowledge.
    try:
        nltk.data.find(corp_path) # Use the internal NLTK check
    except LookupError: # This is the more general error for data not found in NLTK
        print(f"NLTK '{corp_name}' ({corp_path}) not found. Downloading...")
        try:
            nltk.download(corp_name)
        except Exception as e: # Catch a more general exception for robustness during download
            print(f"Error downloading {corp_name}: {e}")
            print(f"Please try running 'python -m nltk.downloader {corp_name}' in your terminal (with venv activated).")
            # You might want to exit or raise a more specific error here if essential data is missing
# --- End NLTK Data Check ---


# Load the English language model for spaCy
try:
    nlp = spacy.load("en_core_web_sm")
    print("spaCy model loaded successfully!")
except OSError:
    print("spaCy model 'en_core_web_sm' not found. Downloading...")
    # This might require administrator privileges or be run directly in terminal
    # Use python -m spacy download en_core_web_sm in your activated venv
    try:
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        print(f"Error downloading spaCy model: {e}")
        print("Please ensure your virtual environment is activated and run 'python -m spacy download en_core_web_sm' manually.")
        print("Exiting as essential model could not be loaded.")
        exit() # Exit if essential model cannot be loaded


def get_synonyms(word):
    """Fetches synonyms for a word using WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ")
            if synonym.lower() != word.lower(): # Avoid adding the word itself
                synonyms.add(synonym)
    return list(synonyms)

def generate_mcq_flashcards(doc_text, num_distractors=3):
    flashcards = []
    doc = nlp(doc_text) # Process the text again within the function

    for sent in doc.sents: # Iterate over each sentence
        sentence = sent.text.strip()
        if not sentence: # Skip empty sentences
            continue

        answer_candidate = None
        # Priority 1: Named Entities (Nouns: GPE, LOC, ORG, PERSON)
        for ent in sent.ents:
            if ent.label_ in ["GPE", "LOC", "ORG", "PERSON", "NORP", "FAC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "PERCENT", "MONEY", "QUANTITY", "ORDINAL"]:
                answer_candidate = ent.text
                break # Take the first suitable entity

        # Priority 2: Noun Chunks (for more general concepts)
        if not answer_candidate:
            for chunk in sent.noun_chunks:
                # Ensure the chunk is reasonably long and not just a common pronoun
                if len(chunk.text.split()) > 1 and len(chunk.text) > 5 and chunk.root.pos_ == 'NOUN':
                    answer_candidate = chunk.text
                    break

        if not answer_candidate:
            continue # No suitable answer found in this sentence

        # Ensure the candidate is not just whitespace or too short
        if not answer_candidate.strip() or len(answer_candidate.strip()) < 3:
            continue

        question_text = sentence.replace(answer_candidate, "______", 1) # Create fill-in-the-blank question
        # Basic check to avoid issues if the answer is a very common word that appears multiple times
        # or if the replacement makes the question nonsensical
        if answer_candidate.lower() in question_text.lower() and "______" not in question_text:
            continue


        # Now, generate distractors for MCQ
        distractors = []
        # Use the last word if it's a phrase, otherwise the whole word
        target_word_for_synonym = answer_candidate.split()[-1] if ' ' in answer_candidate else answer_candidate
        possible_distractors = get_synonyms(target_word_for_synonym)

        # Fallback if not enough synonyms: Find other nouns/entities in the text
        if len(possible_distractors) < num_distractors:
             temp_distractors_from_text = []
             # Collect all named entities (excluding the answer itself)
             for other_ent in doc.ents: # Search whole doc for variety
                 if other_ent.text.lower() != answer_candidate.lower() and len(other_ent.text.split()) < 4: # Keep distractors concise
                     temp_distractors_from_text.append(other_ent.text)
             # Collect common nouns (excluding the answer itself)
             for token in doc:
                 if token.pos_ == 'NOUN' and token.text.lower() != answer_candidate.lower() and len(token.text) > 2:
                     temp_distractors_from_text.append(token.text)

             random.shuffle(temp_distractors_from_text)
             for d_text in temp_distractors_from_text:
                 if d_text not in possible_distractors:
                     possible_distractors.append(d_text)
                     if len(possible_distractors) >= num_distractors * 2: # Get more than needed initially
                         break


        # Select unique distractors, ensuring they are not the answer itself
        final_distractors = []
        for d in possible_distractors:
            if d.lower() != answer_candidate.lower() and d.lower() not in [dist.lower() for dist in final_distractors]:
                final_distractors.append(d)
                if len(final_distractors) >= num_distractors:
                    break

        # If we still don't have enough distractors, pad with dummy ones (less ideal but prevents errors)
        while len(final_distractors) < num_distractors:
            final_distractors.append(f"Option {len(final_distractors) + 1}") # Generic placeholders


        # Combine answer and selected distractors, then shuffle for options
        options = [answer_candidate] + random.sample(final_distractors, min(len(final_distractors), num_distractors))
        random.shuffle(options)

        flashcards.append({
            "type": "mcq",
            "question": question_text.strip(),
            "answer": answer_candidate.strip(),
            "options": options
        })

    return flashcards

# --- Example Usage (will run when the script is executed directly) ---
if __name__ == "__main__":
    example_text = """
    The capital city of France is Paris. Paris is famous for the Eiffel Tower and the Louvre Museum. The River Seine flows through Paris. France is located in Western Europe. The year 1789 marked the beginning of the French Revolution.
    """
    print("Processing text to generate flashcards...")
    mcq_cards = generate_mcq_flashcards(example_text)

    print("\n--- Generated MCQ Flashcards ---")
    if mcq_cards:
        for i, card in enumerate(mcq_cards):
            print(f"Flashcard {i+1} (MCQ):")
            print(f"  Q: {card['question']}")
            print(f"  Options: {', '.join(card['options'])}")
            print(f"  A: {card['answer']}\n")
    else:
        print("No MCQ flashcards could be generated from the provided text.")