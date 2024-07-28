# -*- coding: utf-8 -*-

if __name__ == '__main__':
    print("Starting latest_local_mistral_llm..")

import os
import sys
import traceback
import pdfkit
import pygal
import pygal.style
import html
import urllib.parse
from collections import defaultdict, OrderedDict

import traceback
import httpcore
import httpx
import time
import pdfkit
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.colors import yellow, black
from reportlab.pdfgen import canvas
import io
from PyPDF2 import PdfReader, PdfWriter


from configparser import ConfigParser
import re
import regex
from collections import defaultdict, Counter, OrderedDict
import hashlib
import string
import base64
from bisect import bisect_right
import statistics

import rapidfuzz.process
import rapidfuzz.fuzz
from fuzzysearch import find_near_matches
from ncls import NCLS

import json_tricks

import tenacity  # for exponential backoff
import tiktoken

from Utilities import init_logging, safeprint, print_exception, loop, debugging, is_dev_machine, data_dir, Timer, \
    read_file, save_file, read_raw, save_raw, read_txt, save_txt, strtobool, async_cached, async_cached_encrypted
from TimeLimit import time_limit

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

if is_dev_machine:
    from pympler import asizeof
letters_regex = regex.compile(r'\p{L}')  # matches unicode letters only, not digits   # regex has better Unicode support than re

def extract_annotations(closed_ended_response):
    annotations = []
    pattern = re.compile(r"\[(.*?)\]: (.*?)- \{(.*?)\}")
    matches = pattern.findall(closed_ended_response)
    for match in matches:
        person, text, labels = match
        labels_list = labels.split(", ") if labels else []
        annotations.append({
            "person": person,
            "text": text.strip(),
            "labels": labels_list
        })
    return annotations

def remove_quotes(text):
    return text.replace("'", "").replace('"', '')

def remove_percent(text):
    if text[-1:] == "%":
        return text[:-1]
    else:
        return text

def rotate_list(list, n):
    return list[n:] + list[:n]

def get_config():
    config = ConfigParser(inline_comment_prefixes=("#", ";"))  # by default, inline comments were not allowed
    config.read('MER.ini')

    extract_message_indexes = strtobool(remove_quotes(config.get("MER", "ExtractMessageIndexes", fallback="false")))
    extract_line_numbers = strtobool(remove_quotes(config.get("MER", "ExtractLineNumbers", fallback="false")))
    do_open_ended_analysis = strtobool(remove_quotes(config.get("MER", "DoOpenEndedAnalysis", fallback="true")))
    do_closed_ended_analysis = strtobool(remove_quotes(config.get("MER", "DoClosedEndedAnalysis", fallback="true")))
    keep_unexpected_labels = strtobool(remove_quotes(config.get("MER", "KeepUnexpectedLabels", fallback="true")))
    chart_type = remove_quotes(config.get("MER", "ChartType", fallback="radar")).strip()
    render_output = strtobool(remove_quotes(config.get("MER", "RenderOutput", fallback="false")))
    create_pdf = strtobool(remove_quotes(config.get("MER", "CreatePdf", fallback="true")))
    treat_entire_text_as_one_person = strtobool(remove_quotes(config.get("MER", "TreatEntireTextAsOnePerson", fallback="false")))  # TODO
    anonymise_names = strtobool(remove_quotes(config.get("MER", "AnonymiseNames", fallback="false")))
    anonymise_numbers = strtobool(remove_quotes(config.get("MER", "AnonymiseNumbers", fallback="false")))
    named_entity_recognition_model = remove_quotes(config.get("MER", "NamedEntityRecognitionModel", fallback="en_core_web_sm")).strip()
    encrypt_cache_data = strtobool(remove_quotes(config.get("MER", "EncryptCacheData", fallback="true")))
    split_messages_by = remove_quotes(config.get("MER", "SplitMessagesBy", fallback=""))  # .strip()
    ignore_incorrectly_assigned_citations = strtobool(remove_quotes(config.get("MER", "IgnoreIncorrectlyAssignedCitations", fallback="false")))
    allow_multiple_citations_per_message = strtobool(remove_quotes(config.get("MER", "AllowMultipleCitationsPerMessage", fallback="true")))
    citation_lookup_time_limit = float(remove_quotes(config.get("MER", "CitationLookupTimeLimit", fallback="0.1")))
    sample_count = int(remove_quotes(config.get("MER", "SampleCount", fallback="1")))
    default_label_treshold_sample_percent = float(remove_percent(remove_quotes(config.get("MER", "DefaultLabelThresholdSamplePercent", fallback="50"))))

    result = {
        "extract_message_indexes": extract_message_indexes,
        "extract_line_numbers": extract_line_numbers,
        "do_open_ended_analysis": do_open_ended_analysis,
        "do_closed_ended_analysis": do_closed_ended_analysis,
        "keep_unexpected_labels": keep_unexpected_labels,
        "chart_type": chart_type,
        "render_output": render_output,
        "create_pdf": create_pdf,
        "treat_entire_text_as_one_person": treat_entire_text_as_one_person,
        "anonymise_names": anonymise_names,
        "anonymise_numbers": anonymise_numbers,
        "named_entity_recognition_model": named_entity_recognition_model,
        "encrypt_cache_data": encrypt_cache_data,
        "split_messages_by": split_messages_by,
        "ignore_incorrectly_assigned_citations": ignore_incorrectly_assigned_citations,
        "allow_multiple_citations_per_message": allow_multiple_citations_per_message,
        "citation_lookup_time_limit": citation_lookup_time_limit,
        "sample_count": sample_count,
        "default_label_treshold_sample_percent": default_label_treshold_sample_percent,
    }

    return result

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

# Get encoding for model
def get_encoding_for_model(model):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        safeprint("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    return encoding

# Remove comments from the text
def remove_comments(text):
    text = re.sub(r"(^|[\r\n]+)\s*#[^\r\n]*", r"\1", text)
    return text


def highlight_text_in_pdf(user_input, annotations, output_pdf_filename):
    # Create a PDF with highlights
    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize=letter)
    can.setFont("Helvetica", 12)

    # Add highlights based on annotations
    for ann in annotations:
        start = user_input.find(ann['text'])
        end = start + len(ann['text'])
        label = ", ".join(ann['labels'])

        # Calculate position (this is a simple example, position calculation might need to be more complex)
        text_width = can.stringWidth(ann['text'], "Helvetica", 12)
        x_position = 1 * inch
        y_position = 10 * inch - (start / len(user_input)) * 9 * inch

        # Draw highlight
        can.setFillColor(yellow)
        can.rect(x_position, y_position, text_width, 12, fill=True, stroke=False)

        # Draw text
        can.setFillColor(black)
        can.drawString(x_position, y_position, ann['text'])

    can.save()

    # Move to the beginning of the StringIO buffer
    packet.seek(0)
    new_pdf = PdfReader(packet)
    existing_pdf = PdfReader(output_pdf_filename)
    output = PdfWriter()

    # Add the "watermark" (which is the new pdf) on the existing page
    page = existing_pdf.pages[0]
    page.merge_page(new_pdf.pages[0])
    output.add_page(page)

    # Finally, write "output" to a real file
    with open(output_pdf_filename, "wb") as outputStream:
        output.write(outputStream)

# Sanitize the input text
def sanitise_input(text):
    text = re.sub(r"[{\[]", "(", text)
    text = re.sub(r"[}\]]", ")", text)
    text = re.sub(r"-{3,}", "--", text)
    return text

# Anonymise uncached data
def anonymise_uncached(user_input, anonymise_names, anonymise_numbers, ner_model):
    with Timer("Loading Spacy"):
        import spacy

    with Timer("Loading Named Entity Recognition model"):
        NER = spacy.load(ner_model)

    entities = NER(user_input)
    letters = string.ascii_uppercase

    next_available_replacement_letter_index = 0
    result = ""
    prev_ent_end = 0
    entities_dict = {}
    reserved_replacement_letter_indexes = set()

    active_replacements = ""
    if anonymise_names:
        active_replacements += "Person|Group|Building|Organisation|Area|Location|Event|Language"
    if anonymise_names and anonymise_numbers:
        active_replacements += "|"
    if anonymise_numbers:
        active_replacements += "Money Amount|Quantity|Number"

    if len(active_replacements) > 0:
        re_matches = re.findall(r"(^|\s)(" + active_replacements + ")(\s+)([" + re.escape(letters) + "])(\s|:|$)", user_input)
        for re_match in re_matches:
            replacement = re_match[2]
            space = re_match[3]
            letter = re_match[4]
            replacement_letter_index = ord(letter) - ord("A")
            reserved_replacement_letter_indexes.add(replacement_letter_index)
            entities_dict[replacement + " " + letter] = replacement_letter_index

    for phase in range(0, 2):
        for word in entities.ents:
            text_original = word.text
            label = word.label_
            start_char = word.start_char
            end_char = word.end_char
            text_normalised = re.sub(r"\s+", " ", text_original)
            if phase == 0 and text_normalised in entities_dict:
                continue
            if label == "PERSON":
                replacement = "Person" if anonymise_names else None
            elif label == "NORP":
                replacement = "Group" if anonymise_names else None
            elif label == "FAC":
                replacement = "Building" if anonymise_names else None
            elif label == "ORG":
                replacement = "Organisation" if anonymise_names else None
            elif label == "GPE":
                replacement = "Area" if anonymise_names else None
            elif label == "LOC":
                replacement = "Location" if anonymise_names else None
            elif label == "PRODUCT":
                replacement = None
            elif label == "EVENT":
                replacement = "Event" if anonymise_names else None
            elif label == "WORK_OF_ART":
                replacement = None
            elif label == "LAW":
                replacement = None
            elif label == "LANGUAGE":
                replacement = "Language" if anonymise_names else None
            elif label == "DATE":
                replacement = None
            elif label == "TIME":
                replacement = None
            elif label == "PERCENT":
                replacement = None
            elif label == "MONEY":
                replacement = "Money Amount" if anonymise_numbers else None
            elif label == "QUANTITY":
                replacement = "Quantity" if anonymise_numbers else None
            elif label == "ORDINAL":
                replacement = None
            elif label == "CARDINAL":
                replacement = (
                    "Number" if
                    anonymise_numbers
                    and len(text_normalised) > 2
                    and re.search(r"(\d|\s)", text_normalised) is not None
                    else None
                )
            else:
                replacement = None

            if phase == 1:
                result += user_input[prev_ent_end:start_char]
                prev_ent_end = end_char

            if replacement is None:
                if phase == 1:
                    result += text_original
            else:
                if phase == 0:
                    if text_normalised not in entities_dict:
                        while next_available_replacement_letter_index in reserved_replacement_letter_indexes:
                            next_available_replacement_letter_index += 1
                        replacement_letter_index = next_available_replacement_letter_index
                        entities_dict[text_normalised] = replacement_letter_index
                        reserved_replacement_letter_indexes.add(replacement_letter_index)
                else:
                    replacement_letter_index = entities_dict[text_normalised]
                    if len(reserved_replacement_letter_indexes) <= len(letters):
                        replacement_letter = letters[replacement_letter_index]
                    else:
                        replacement_letter = str(replacement_letter_index + 1)
                    result += replacement + " " + replacement_letter

        result += user_input[prev_ent_end:]

    return result

async def anonymise(config, user_input, anonymise_names, anonymise_numbers, ner_model, enable_cache=True):
    user_input = re.sub(r"[^\S\r\n]+", " ", user_input)
    encrypt_cache_data = config["encrypt_cache_data"]
    if encrypt_cache_data:
        result = await async_cached_encrypted(1 if enable_cache else None, anonymise_uncached, user_input, anonymise_names, anonymise_numbers, ner_model)
    else:
        result = await async_cached(1 if enable_cache else None, anonymise_uncached, user_input, anonymise_names, anonymise_numbers, ner_model)
    return result


def render_highlights_uncached(user_input, closed_ended_response):
    from spacy import displacy

    annotations = extract_annotations(closed_ended_response)

    # Create highlights using spacy's displacy
    highlights = []
    for ann in annotations:
        highlights.append({
            "start": user_input.find(ann['text']),
            "end": user_input.find(ann['text']) + len(ann['text']),
            "label": ", ".join(ann['labels'])
        })

    entities = [{"start": hl["start"], "end": hl["end"], "label": hl["label"]} for hl in highlights]

    doc = {
        "text": user_input,
        "ents": entities,
        "title": None
    }

    options = {"ents": list(set(hl["label"] for hl in highlights)), "colors": {"LABEL": "yellow"}}
    html = displacy.render(doc, style="ent", manual=True, options=options)

    return html

# Wrapper function to support caching if needed
async def render_highlights(config, user_input, closed_ended_response, enable_cache=True):
    encrypt_cache_data = config["encrypt_cache_data"]
    if encrypt_cache_data:
        result = await async_cached_encrypted(1 if enable_cache else None, render_highlights_uncached, user_input, closed_ended_response)
    else:
        result = await async_cached(1 if enable_cache else None, render_highlights_uncached, user_input, closed_ended_response)
    return result

def parse_labels(all_labels_as_text):
    labels_list = []
    lines = all_labels_as_text.splitlines(keepends=False)
    for line in lines:
        line = line.strip()
        if line[:1] == "-":
            line = line[1:].strip()
        line = sanitise_input(line)
        line = re.sub(r"[.,:;]+", "/", line).strip()
        if len(line) == 0:
            continue
        labels_list.append(line)
    all_labels_as_text = "\n".join("- " + x for x in labels_list)
    return (labels_list, all_labels_as_text)

def split_text_into_chunks_worker(encoding, paragraphs, paragraph_token_counts, separator, separator_token_count, max_tokens_per_chunk, overlap_chunks_at_least_halfway=False):
    chunks = []
    current_chunk = []
    current_chunk_token_count = 0

    for paragraph, paragraph_token_count in zip(paragraphs, paragraph_token_counts):
        if current_chunk_token_count > 0:
            if current_chunk_token_count + separator_token_count + paragraph_token_count <= max_tokens_per_chunk:
                current_chunk_token_count += separator_token_count + paragraph_token_count
                current_chunk.append(separator)
                current_chunk.append(paragraph)
                continue
            else:
                chunks.append((current_chunk, current_chunk_token_count))
                if overlap_chunks_at_least_halfway:
                    # Retain the last part of the chunk to overlap halfway
                    overlap_start_index = len(current_chunk) // 2
                    current_chunk = current_chunk[overlap_start_index:]
                    current_chunk_token_count = sum(len(encoding.encode(p)) for p in current_chunk)
                else:
                    current_chunk = []
                    current_chunk_token_count = 0

        if paragraph_token_count <= max_tokens_per_chunk:
            current_chunk_token_count = paragraph_token_count
            current_chunk.append(paragraph)
        else:
            assert False

    if current_chunk_token_count > 0:
        chunks.append((current_chunk, current_chunk_token_count))

    return chunks


def split_text_into_chunks(encoding, text, separator, max_tokens_per_chunk):
    paragraphs = text.split(separator)
    chunks = []
    current_chunk = []
    current_chunk_token_count = 0

    for paragraph in paragraphs:
        paragraph_token_count = len(encoding.encode(paragraph))
        if current_chunk_token_count + paragraph_token_count + len(encoding.encode(separator)) > max_tokens_per_chunk:
            chunks.append(separator.join(current_chunk))
            current_chunk = []
            current_chunk_token_count = 0

        current_chunk.append(paragraph)
        current_chunk_token_count += paragraph_token_count + len(encoding.encode(separator))

    if current_chunk:
        chunks.append(separator.join(current_chunk))

    return chunks

async def recogniser_process_chunk(user_input, config, instructions, encoding, do_open_ended_analysis=None, do_closed_ended_analysis=None, extract_message_indexes=None, extract_line_numbers=None, sample_index=0):
    open_ended_system_instruction = instructions["open_ended_system_instruction"]
    extract_names_of_participants_system_instruction = instructions["extract_names_of_participants_system_instruction"]
    labels_list = instructions["labels_list"]
    ignored_labels_list = instructions["ignored_labels_list"]
    continuation_request = instructions["continuation_request"]
    closed_ended_system_instruction_with_labels = instructions["closed_ended_system_instruction_with_labels"]

    if do_open_ended_analysis:
        open_ended_response = await process_chunk(user_input, "mistral")
    else:
        open_ended_response = None

    if do_closed_ended_analysis:
        closed_ended_response = await process_chunk(closed_ended_system_instruction_with_labels + "\n---\n" + user_input, "mistral")
        expressions_tuples = []
        detected_persons = set()

        if extract_message_indexes:
            names_of_participants_response = await process_chunk(extract_names_of_participants_system_instruction, "mistral")
            re_matches = re.findall(r"[\r\n]+\[?([^\]\n]*)\]?", "\n" + names_of_participants_response)
            for re_match in re_matches:
                detected_persons.add(re_match)

        re_matches = re.findall(r"[\r\n]+\[(.*)\]:(.*)\{(.*)\}", "\n" + closed_ended_response)
        for re_match in re_matches:
            (person, citation, labels) = re_match
            citation = citation.strip()
            if citation[-1] == "-":
                citation = citation[0:-1].strip()
            detected_persons.add(person)
            labels = [x.strip() for x in labels.split(",") if x.strip()]
            filtered_labels = [label for label in labels if label not in ignored_labels_list]
            if filtered_labels and citation.strip():
                expressions_tuples.append((person, citation, filtered_labels))

        labels_list.sort()
        if extract_line_numbers:
            line_start_char_positions = [0]
            for re_match in re.finditer(r"\n", user_input):
                line_start_char_positions.append(re_match.start() + 1)
            num_lines = len(line_start_char_positions)
        else:
            num_lines = None

        person_messages = {person: [] for person in detected_persons}
        overall_message_indexes = {person: {} for person in detected_persons}
        message_line_numbers = {person: {} for person in detected_persons}
        start_char_to_person_message_index_dict = {}
        person_message_spans = {person: [] for person in detected_persons}

        split_messages_by = config["split_messages_by"]
        split_messages_by_newline = (split_messages_by == "")

        for person in detected_persons:
            pattern = re.escape(person) + r":(.*?)[\r\n]+" + re.escape(split_messages_by) if not split_messages_by_newline else re.escape(person) + r":(.*)"
            re_matches = re.finditer(r"[\r\n]+" + pattern, "\n" + user_input + "\n" + split_messages_by if not split_messages_by_newline else "\n" + user_input)
            for re_match in re_matches:
                message = re_match.group(1).strip()
                start_char = re_match.start(1)
                end_char = start_char + len(message)
                start_char_to_person_message_index_dict[start_char] = (person, len(person_messages[person]))
                person_messages[person].append(message)
                person_message_spans[person].append((start_char, end_char))

        num_messages = len(start_char_to_person_message_index_dict)
        start_char_to_person_message_index_dict = OrderedDict(sorted(start_char_to_person_message_index_dict.items()))

        for overall_message_index, entry in enumerate(start_char_to_person_message_index_dict.values()):
            person, person_message_index = entry
            overall_message_indexes[person][person_message_index] = overall_message_index

        totals = defaultdict(lambda: OrderedDict([(label, 0) for label in labels_list]))
        expression_dicts = []
        already_labelled_message_parts = {}

        ignore_incorrectly_assigned_citations = config["ignore_incorrectly_assigned_citations"]
        allow_multiple_citations_per_message = config["allow_multiple_citations_per_message"]

        for tuple_index, (person, citation, labels) in enumerate(expressions_tuples):
            nearest_message_similarity = 0
            nearest_message_is_partial_match = None
            nearest_person = None
            nearest_message = None
            nearest_person_message_index = None

            for person2 in detected_persons:
                curr_person_messages = person_messages.get(person2)
                if curr_person_messages:
                    match = rapidfuzz.process.extractOne(citation, curr_person_messages, scorer=rapidfuzz.fuzz.partial_ratio)
                    if match:
                        person2_nearest_message, similarity, person2_message_index = match
                        similarity_is_partial_match = True
                        if len(citation) > len(person2_nearest_message):
                            match = rapidfuzz.process.extractOne(citation, curr_person_messages, scorer=rapidfuzz.fuzz.ratio)
                            if match:
                                person2_nearest_message, similarity, person2_message_index = match
                                similarity_is_partial_match = False
                        if similarity > nearest_message_similarity:
                            nearest_message_similarity = similarity
                            nearest_message_is_partial_match = similarity_is_partial_match
                            nearest_person = person2
                            nearest_message = person2_nearest_message
                            nearest_person_message_index = person2_message_index

            if nearest_person and nearest_person != person:
                if ignore_incorrectly_assigned_citations:
                    continue
                person = nearest_person

            if not allow_multiple_citations_per_message and nearest_message in already_labelled_message_parts:
                already_labelled_message_parts[nearest_message]["labels"].extend(labels)
                continue

            if nearest_person_message_index is not None:
                citation_in_nearest_message = nearest_message
                person_message_start_char = person_message_spans[person][nearest_person_message_index][0]
                start_char = person_message_start_char
                end_char = start_char + len(nearest_message)

                entry = {
                    "person": person,
                    "start_char": start_char,
                    "end_char": end_char,
                    "text": citation_in_nearest_message,
                    "labels": labels,
                }

                if extract_message_indexes:
                    entry["message_index"] = overall_message_indexes[person][nearest_person_message_index]

                if extract_line_numbers:
                    entry["line_number"] = bisect_right(line_start_char_positions, start_char) + 1

                if not allow_multiple_citations_per_message:
                    already_labelled_message_parts[nearest_message] = entry

                expression_dicts.append(entry)

        expression_dicts.sort(key=lambda entry: entry["start_char"])
        user_input_len = len(user_input)

    else:
        closed_ended_response = None
        totals = None
        expression_dicts = None
        user_input_len = None

    if expression_dicts:
        for entry in expression_dicts:
            entry["labels"] = list(set(entry["labels"]))
            entry["labels"].sort()
            person = entry["person"]
            for label in entry["labels"]:
                if label not in totals[person]:
                    totals[person][label] = 0
                totals[person][label] += 1

    return expression_dicts, totals, closed_ended_response, open_ended_response, num_messages, num_lines, user_input_len


async def recogniser(do_open_ended_analysis=None, do_closed_ended_analysis=None, extract_message_indexes=None, extract_line_numbers=None, argv=None):
    argv = argv if argv else sys.argv
    config = get_config()
    if do_open_ended_analysis is None:
        do_open_ended_analysis = config["do_open_ended_analysis"]
    if do_closed_ended_analysis is None:
        do_closed_ended_analysis = config["do_closed_ended_analysis"]
    if extract_message_indexes is None:
        extract_message_indexes = config["extract_message_indexes"]
    if extract_line_numbers is None:
        extract_line_numbers = config["extract_line_numbers"]

    labels_filename = argv[3] if len(argv) > 3 else None
    if labels_filename:
        labels_filename = os.path.join("..", labels_filename)
    else:
        labels_filename = "new_labels.txt"

    ignored_labels_filename = argv[4] if len(argv) > 4 else None
    if ignored_labels_filename:
        ignored_labels_filename = os.path.join("..", ignored_labels_filename)
    else:
        ignored_labels_filename = "ignored_labels.txt"

    closed_ended_system_instruction = (await read_txt("closed_ended_system_instruction.txt", quiet=True)).lstrip()
    open_ended_system_instruction = (await read_txt("open_ended_system_instruction.txt", quiet=True)).lstrip()
    extract_names_of_participants_system_instruction = (await read_txt("extract_names_of_participants_system_instruction.txt", quiet=True)).lstrip()
    all_labels_as_text = (await read_txt(labels_filename, quiet=True)).strip()
    all_ignored_labels_as_text = (await read_txt(ignored_labels_filename, quiet=True)).strip()
    continuation_request = (await read_txt("continuation_request.txt", quiet=True)).strip()

    closed_ended_system_instruction_with_labels = closed_ended_system_instruction.replace("%labels%", all_labels_as_text)

    input_filename = argv[1] if len(argv) > 1 else None
    if input_filename:
        input_filename = os.path.join("..", input_filename)
        using_user_input_filename = True
    else:
        input_filename = "test_input12.txt"
        using_user_input_filename = False

    user_input = await read_txt(input_filename, quiet=True)
    user_input = remove_comments(user_input)

    anonymise_names = config.get("anonymise_names")
    anonymise_numbers = config.get("anonymise_numbers")
    ner_model = config.get("named_entity_recognition_model")

    if anonymise_names or anonymise_numbers:
        user_input = await anonymise(config, user_input, anonymise_names, anonymise_numbers, ner_model)

    user_input = sanitise_input(user_input)

    (labels_list, all_labels_as_text) = parse_labels(all_labels_as_text)
    (ignored_labels_list, all_ignored_labels_as_text) = parse_labels(all_ignored_labels_as_text)

    instructions = {
        "open_ended_system_instruction": open_ended_system_instruction,
        "extract_names_of_participants_system_instruction": extract_names_of_participants_system_instruction,
        "labels_list": labels_list,
        "ignored_labels_list": ignored_labels_list,
        "continuation_request": continuation_request,
        "closed_ended_system_instruction_with_labels": closed_ended_system_instruction_with_labels,
    }

    separator = "\n"
    encoding = get_encoding_for_model("mistral")
    model_max_tokens = 2048
    reserve_tokens = 100
    max_tokens_per_chunk = model_max_tokens - reserve_tokens

    chunks = split_text_into_chunks(encoding, user_input, separator, max_tokens_per_chunk)

    sample_count = config["sample_count"]
    default_label_treshold_sample_percent = config["default_label_treshold_sample_percent"]
    expression_dicts_samples = []

    open_ended_responses = []
    closed_ended_responses = []

    for sample_index in range(0, sample_count):
        safeprint(f"Collecting sample {(sample_index + 1)} / {sample_count}")

        chunk_analysis_results = []
        for index, chunk_text in enumerate(chunks):
            with Timer(f"Analysing chunk {(index + 1)} / {len(chunks)} of sample {(sample_index + 1)} / {sample_count}"):
                chunk_analysis_result = await recogniser_process_chunk(chunk_text, config, instructions, encoding, (do_open_ended_analysis if sample_index == 0 else False), do_closed_ended_analysis, extract_message_indexes, extract_line_numbers, sample_index)
                chunk_analysis_results.append(chunk_analysis_result)

        if do_closed_ended_analysis:
            totals = defaultdict(lambda: OrderedDict([(label, 0) for label in labels_list]))
            expression_dicts = []
        else:
            totals = None
            expression_dicts = None

        prev_chunks_lengths_sum = 0
        prev_chunks_messages_count = 0
        prev_chunks_lines_count = 0

        for result in chunk_analysis_results:
            (chunk_expression_dicts, chunk_totals, chunk_closed_ended_response, chunk_open_ended_response, chunk_num_messages, chunk_num_lines, chunk_user_input_len) = result

            if do_closed_ended_analysis:
                for expression_dict in chunk_expression_dicts:
                    expression_dict["start_char"] += prev_chunks_lengths_sum
                    expression_dict["end_char"] += prev_chunks_lengths_sum
                    if extract_message_indexes:
                        expression_dict["message_index"] += prev_chunks_messages_count
                    if extract_line_numbers:
                        expression_dict["line_number"] += prev_chunks_lines_count
                    expression_dicts.append(expression_dict)

            prev_chunks_lengths_sum += chunk_user_input_len + len(separator)

            if extract_message_indexes:
                prev_chunks_messages_count += chunk_num_messages

            if extract_line_numbers:
                prev_chunks_lines_count += chunk_num_lines

            for person, counts in chunk_totals.items():
                for label, count in counts.items():
                    if label not in totals[person]:
                        totals[person][label] = 0
                    totals[person][label] += count

            if do_closed_ended_analysis:
                closed_ended_responses.append(chunk_closed_ended_response)
            if do_open_ended_analysis and sample_index == 0:
                open_ended_responses.append(chunk_open_ended_response)

        if sample_index == 0:
            if do_open_ended_analysis:
                open_ended_response = "\n\n".join(open_ended_responses)
            else:
                open_ended_response = None

        if do_closed_ended_analysis:
            closed_ended_response = "\n\n".join(closed_ended_responses)
        else:
            closed_ended_response = None

        if do_closed_ended_analysis:
            expression_dicts_samples.append(expression_dicts)

    if not do_closed_ended_analysis:
        filtered_expression_dicts = None
        filtered_totals = None
        filtered_unexpected_labels = None
        filtered_aggregated_unused_labels = None
    elif sample_count == 1:
        filtered_expression_dicts = expression_dicts_samples[0]
        for entry in filtered_expression_dicts:
            entry["labels"] = OrderedDict([(label, 100) for label in entry["labels"]])
        filtered_totals = totals
    else:
        with Timer("Filtering labels based on confidence level"):
            starts = []
            ends = []
            intervals = []
            unique_start_points = set()
            unique_end_points = set()
            unique_points = set()
            for expression_dicts in expression_dicts_samples:
                for expression_dict in expression_dicts:
                    start_char = expression_dict["start_char"]
                    end_char = expression_dict["end_char"]
                    unique_start_points.add(start_char)
                    unique_end_points.add(end_char)
                    unique_points.add(start_char)
                    unique_points.add(end_char)
                    starts.append(start_char)
                    ends.append(end_char)
                    intervals.append(expression_dict)
            if len(intervals) > 0:
                ncls = NCLS(starts, ends, range(0, len(intervals)))
            unique_start_points = list(unique_start_points)
            unique_end_points = list(unique_end_points)
            unique_points = list(unique_points)
            unique_start_points.sort()
            unique_end_points.sort()
            unique_points.sort()
            default_label_treshold_float = sample_count * default_label_treshold_sample_percent / 100
            filtered_expression_dicts = []
            prev_point = 0
            for point in unique_points:
                spans = ncls.find_overlap(prev_point, point)
                intervals_per_span_range_dict = defaultdict(list)
                for span in spans:
                    (start_char, end_char, interval_id) = span
                    interval = intervals[interval_id]
                    if prev_point != start_char or point != end_char:
                        qqq = True
                    span_start = max(prev_point, start_char)
                    span_end = min(point, end_char)
                    if span_end - span_start <= 1:
                        continue
                    key = (span_start, span_end)
                    intervals_per_span_range_dict[key].append(interval)
                for (span_start, span_end), intervals_per_span_range in intervals_per_span_range_dict.items():
                    label_counts_in_span_range = Counter()
                    persons = set()
                    for interval in intervals_per_span_range:
                        for label in interval["labels"]:
                            label_counts_in_span_range[label] += 1
                        person = interval["person"]
                        persons.add(person)
                    assert(len(persons) == 1)
                    filtered_labels = OrderedDict()
                    for label, count in label_counts_in_span_range.items():
                        if default_label_treshold_sample_percent < 100:
                            if count > default_label_treshold_float:
                                filtered_labels[label] = count / sample_count * 100
                        else:
                            if count == sample_count:
                                filtered_labels[label] = 100
                    if len(filtered_labels) > 0:
                        text = user_input[span_start:span_end]
                        filtered_labels = OrderedDict(sorted(filtered_labels.items()))
                        if not letters_regex.search(text):
                            continue
                        entry = {
                            "person": person,
                            "start_char": span_start,
                            "end_char": span_end,
                            "text": text,
                            "labels": filtered_labels,
                        }
                        if extract_message_indexes:
                            entry.update({
                                "message_index": intervals_per_span_range[0]["message_index"]
                            })
                        if extract_line_numbers:
                            entry.update({
                                "line_number": intervals_per_span_range[0]["line_number"]
                            })
                        filtered_expression_dicts.append(entry)
                prev_point = point
            filtered_expression_dicts.sort(key=lambda entry: entry["start_char"])
            filtered_totals = defaultdict(lambda: OrderedDict([(label, 0) for label in labels_list]))
            for entry in filtered_expression_dicts:
                person = entry["person"]
                labels = entry["labels"]
                for label, percent in labels.items():
                    if label not in filtered_totals[person]:
                        filtered_totals[person][label] = 0
                    filtered_totals[person][label] += 1

    if not do_closed_ended_analysis:
        filtered_unexpected_labels = None
    else:
        keep_unexpected_labels = config.get("keep_unexpected_labels")
        filtered_unexpected_labels = set()
        filtered_aggregated_unused_labels = list(labels_list)
        filtered_aggregated_unused_labels_set = set(labels_list)
        for entry in filtered_expression_dicts:
            labels = entry["labels"]
            if not keep_unexpected_labels:
                filtered_labels = OrderedDict()
            for label, percent in labels.items():
                if label not in labels_list:
                    filtered_unexpected_labels.add(label)
                    if not keep_unexpected_labels:
                        pass
                else:
                    if not keep_unexpected_labels:
                        filtered_labels[label] = percent
                if label in filtered_aggregated_unused_labels_set:
                    filtered_aggregated_unused_labels.remove(label)
                    filtered_aggregated_unused_labels_set.remove(label)
            if not keep_unexpected_labels:
                entry["labels"] = filtered_labels
        filtered_unexpected_labels = list(filtered_unexpected_labels)
        filtered_unexpected_labels.sort()
        for (person, person_counts) in filtered_totals.items():
            filtered_totals[person] = OrderedDict([(key, value) for (key, value) in person_counts.items() if value > 0])
        del expression_dicts
        del totals

    analysis_response = {
        "error_code": 0,
        "error_msg": "",
        "sanitised_text": user_input,
        "expressions": filtered_expression_dicts,
        "counts": filtered_totals,
        "unexpected_labels": filtered_unexpected_labels,
        "unused_labels": filtered_aggregated_unused_labels,
        "raw_expressions_labeling_response": closed_ended_response,
        "qualitative_evaluation": open_ended_response
    }

    response_json = json_tricks.dumps(analysis_response, indent=2)

    response_filename = argv[2] if len(argv) > 2 else None
    if response_filename:
        response_filename = os.path.join("..", response_filename)
    else:
        response_filename = os.path.splitext(input_filename)[0] + "_evaluation.json" if using_user_input_filename else "test_evaluation.json"

    await save_txt(response_filename, response_json, quiet=True, make_backup=True, append=False)

    safeprint("Analysis done.")

    # Print the closed_ended_response
    print("\nClosed-ended response:\n")
    print(closed_ended_response)

    # Save the closed_ended_response to a file
    closed_ended_response_filename = os.path.splitext(response_filename)[0] + "_closed_ended_response.txt"
    await save_txt(closed_ended_response_filename, closed_ended_response, quiet=True, make_backup=True, append=False)

    # Render and save highlights in PDF based on closed_ended_response
    annotations = extract_annotations(closed_ended_response)
    pdf_output_filename = os.path.splitext(response_filename)[0] + "_output_with_highlights.pdf"

    render_output = config.get("render_output")
    if render_output:
        chart_type = config.get("chart_type")
        title = "Manipulative Expression Recognition (MER)"
        style = pygal.style.DefaultStyle(
            foreground="rgba(0, 0, 0, .87)",
            foreground_strong="rgba(128, 128, 128, 1)",
            guide_stroke_color="#404040",
            major_guide_stroke_color="#808080",
            stroke_width=2,
        )
        if chart_type == "radar":
            chart = pygal.Radar(style=style, order_min=0)
            reverse_labels_order = True
            shift_labels_order_left = True
        elif chart_type == "vbar":
            chart = pygal.Bar(style=style, order_min=0)
            reverse_labels_order = False
            shift_labels_order_left = False
        elif chart_type == "hbar":
            chart = pygal.HorizontalBar(style=style, order_min=0)
            reverse_labels_order = True
            shift_labels_order_left = False
        else:
            chart = None
            reverse_labels_order = False
            shift_labels_order_left = False
        nonzero_labels_list = []
        if True:
            for label in labels_list:
                for (person, person_counts) in filtered_totals.items():
                    count = person_counts.get(label, 0)
                    if count > 0:
                        nonzero_labels_list.append(label)
                        break
        if chart:
            x_labels = list(nonzero_labels_list)
            if reverse_labels_order:
                x_labels.reverse()
            if shift_labels_order_left:
                x_labels = rotate_list(x_labels, -1)
            chart.title = title
            chart.x_labels = x_labels
            if True:
                for (person, person_counts) in filtered_totals.items():
                    series = [person_counts.get(label, 0) for label in x_labels]
                    chart.add(person, series)
            svg = chart.render()
            response_svg_filename = os.path.splitext(response_filename)[0] + ".svg" if using_user_input_filename else "test_evaluation.svg"
            await save_raw(response_svg_filename, svg, quiet=True, make_backup=True, append=False)
        response_html_filename = os.path.splitext(response_filename)[0] + ".html" if using_user_input_filename else "test_evaluation.html"
        highlights_html = await render_highlights(config, user_input, closed_ended_response)
        output_html = get_full_html(title, svg, highlights_html, open_ended_responses, nonzero_labels_list, response_svg_filename, response_html_filename, for_pdfkit=False)
        await save_txt(response_html_filename, output_html, quiet=True, make_backup=True, append=False)
        create_pdf = config.get("create_pdf")
        if create_pdf:
            pdfkit_html = get_full_html(title, svg, highlights_html, open_ended_responses, nonzero_labels_list, response_svg_filename, response_html_filename, for_pdfkit=True)
            render_html_to_pdf(pdfkit_html, pdf_output_filename)

    highlight_text_in_pdf(user_input, annotations, pdf_output_filename)

    return analysis_response


def get_full_html(title, svg, highlights_html, open_ended_responses, nonzero_labels_list, response_svg_filename, response_html_filename, for_pdfkit=False):
    result = (
        '<html>'
        + '\n<head>'
        + '\n<meta charset="utf-8">'
        + '\n<title>' + html.escape(title) + '</title>'
        + """\n<style>
          .entities {
            line-height: 1.5 !important;
          }
          mark {
            line-height: 2 !important;
            background: yellow !important;
          }
          mark span {
            background: orange !important;
            padding: 0.5em;
          }
          .graph object, .graph svg, .graph img {
            max-height: 75vh;
          }
          .entity {
            padding-top: 0.325em !important;
          }
          mark span {
            vertical-align: initial !important;
          }
        </style>"""
        + '\n</head>'
        + '\n<body>'
        + (
            (
                (
                    ('\n<div class="graph">' + svg.decode('utf8', 'ignore').replace('<svg', '<svg width="1000" height="750"') + '</div>')
                    if for_pdfkit else
                    (
                        '\n<div class="graph"><object data="'
                        + urllib.parse.quote_plus(
                            os.path.relpath(
                                response_svg_filename,
                                os.path.dirname(response_html_filename)
                            ).replace("\\", "/"),
                            safe="/"
                        )
                        + '" type="image/svg+xml"></object></div>'
                    )
                )
                if svg else
                ''
            )
            if len(nonzero_labels_list) > 0 else
            '\n<div style="font: bold 1em Arial;">No manipulative expressions detected.</div>\n<br><br><br>'
        )
        + '\n<div style="font: bold 1em Arial;">Qualitative summary:</div><br>'
        + '\n<div style="font: 1em Arial;">'
        + '\n' + "<br><br>".join(open_ended_responses)
        + '\n</div>'
        + '\n<br><br><br>'
        + '\n<div style="font: bold 1em Arial;">Labelled input:</div><br>'
        + '\n<div style="font: 1em Arial;">'
        + '\n' + highlights_html
        + '\n</div>\n</body>\n</html>'
    )
    return result

def render_html_to_pdf(html_content, output_pdf_filename):
    try:
        pdfkit.from_string(html_content, output_pdf_filename)
    except Exception as ex:
        print("Error creating pdf. Is wkhtmltopdf utility installed?")
        print(str(ex))


if __name__ == '__main__':
    loop.run_until_complete(recogniser())
