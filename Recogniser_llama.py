import os
import asyncio
import aiofiles
import re
import spacy

# Change to script directory
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Read a text file asynchronously
async def read_txt(filename):
    async with aiofiles.open(filename, mode='r', encoding='utf-8') as f:
        return await f.read()

# Remove comments from the text
def remove_comments(text):
    return re.sub(r"(^|[\r\n]+)\s*#[^\r\n]*", r"\1", text)

# Sanitize input by replacing specific characters
def sanitise_input(text):
    text = re.sub(r"[{\[]", "(", text)
    text = re.sub(r"[}\]]", ")", text)
    text = re.sub(r"-{3,}", "--", text)
    return text

# Anonymize names and numbers in the text
def anonymise_text(user_input, anonymise_names, anonymise_numbers, ner_model):
    try:
        NER = spacy.load(ner_model)
    except IOError:
        spacy.cli.download(ner_model)
        NER = spacy.load(ner_model)

    entities = NER(user_input)
    result = user_input

    for entity in entities.ents:
        if anonymise_names and entity.label_ == "PERSON":
            result = result.replace(entity.text, "Person")
        if anonymise_numbers and entity.label_ == "MONEY":
            result = result.replace(entity.text, "Money Amount")

    return result

# Split text into chunks
def split_text_into_chunks(text, max_tokens):
    words = text.split()
    chunks = [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]
    return chunks

# Process each chunk with the LLaMA model using Ollama
async def process_chunk(chunk, model_name):
    import ollama
    stream = ollama.chat(
        model=model_name,
        messages=[{'role': 'user', 'content': chunk}],
        stream=True,
    )
    response = ""
    for chunk in stream:
        response += chunk['message']['content']
    return response

# Main recogniser function
async def recogniser():
    # Configuration parameters
    input_filename = 'data/test_input.txt'
    anonymise_names = True
    anonymise_numbers = True
    ner_model = "en_core_web_sm"
    llm_model = "llama3"
    max_tokens_per_chunk = 512

    user_input = await read_txt(input_filename)
    user_input = remove_comments(user_input)
    user_input = sanitise_input(user_input)

    if anonymise_names or anonymise_numbers:
        user_input = anonymise_text(user_input, anonymise_names, anonymise_numbers, ner_model)

    chunks = split_text_into_chunks(user_input, max_tokens_per_chunk)

    # Process each chunk and collect responses
    tasks = [process_chunk(chunk, llm_model) for chunk in chunks]
    results = await asyncio.gather(*tasks)

    # Save results to a file
    async with aiofiles.open("output.html", mode='w', encoding='utf-8') as f:
        await f.write("<html><body>")
        for result in results:
            await f.write(f"<p>{result}</p>")
        await f.write("</body></html>")

    print("Analysis completed.")

# Run the recogniser function
if __name__ == "__main__":
    asyncio.run(recogniser())
