Manipulative Expression Recognition (MER) and LLM Manipulativeness Benchmark
Introduction
In the rapidly evolving world of artificial intelligence, the rise of large language models like OpenAI's GPT series has brought about profound shifts in digital communication. These models, capable of generating human-like text, have widespread applications ranging from content creation to customer service. As of 2023, their influence is undeniable and pervasive, extending even into areas such as personalized education and virtual companionship.

However, with great capability comes an inherent need for responsibility and scrutiny. Ensuring alignment with human values and understanding the





give me for a redme file format like i gave u 
ChatGPT
Manipulative Expression Recognition (MER) and LLM Manipulativeness Benchmark
Introduction
In the rapidly evolving world of artificial intelligence, the rise of large language models like OpenAI's GPT series has brought about profound shifts in digital communication. These models, capable of generating human-like text, have widespread applications ranging from content creation to customer service. As of 2023, their influence is undeniable and pervasive, extending even into areas such as personalized education and virtual companionship.

However, with great capability comes an inherent need for responsibility and scrutiny. Ensuring alignment with human values and understanding the underlying communicative tendencies of these models is paramount. Specifically, evaluating and benchmarking their potential for manipulative expressions becomes a crucial task.

At the same time, in human communication, manipulative behaviors can significantly impact interpersonal relationships, business negotiations, politics, and many other areas of life. These behaviors often go unnoticed or unrecognized, leaving victims of manipulation without the support and tools needed to defend themselves.

It is against this backdrop that "Manipulative Expression Recognition (MER) and LLM Manipulativeness Benchmark" comes into the picture. Unlike various existing fact-checking software, this software focuses on psychological communication style detection and labeling.

Functionality
MER is designed to provide a comprehensive solution to the challenges mentioned above. It is a software library that allows users to upload transcripts of conversations or individual messages. The software then analyzes the text, applying labels that indicate potential manipulative communication styles.

This tool offers two main use cases:

Large Language Model Evaluation: As more sophisticated models continue to emerge, the need for tools to measure their alignment with human values grows concurrently. MER enables developers and researchers to evaluate and benchmark language model outputs for potential manipulative expressions. This can help inform adjustments and improvements to these models, promoting transparency, safety, and ethical considerations in AI development.

Human Communication Analysis: The application of MER extends beyond AI. By analyzing human-to-human conversations, MER can

no its not MER Manipulative-Communication-Detection-for-Cybersecurity-and-Anti-Scam-MCDCAS- 
ChatGPT
Manipulative Communication Detection for Cybersecurity and Anti-Scam (MCDCAS)
Introduction
In the rapidly evolving world of artificial intelligence, the rise of large language models like OpenAI's GPT series has brought about profound shifts in digital communication. These models, capable of generating human-like text, have widespread applications ranging from content creation to customer service. As of 2023, their influence is undeniable and pervasive, extending even into areas such as personalized education and virtual companionship.

However, with great capability comes an inherent need for responsibility and scrutiny. Ensuring alignment with human values and understanding the underlying communicative tendencies of these models is paramount. Specifically, evaluating and benchmarking their potential for manipulative expressions becomes a crucial task.

At the same time, in human communication, manipulative behaviors can significantly impact interpersonal relationships, business negotiations, politics, and many other areas of life. These behaviors often go unnoticed or unrecognized, leaving victims of manipulation without the support and tools needed to defend themselves.

It is against this backdrop that the "Manipulative Communication Detection for Cybersecurity and Anti-Scam (MCDCAS)" software comes into the picture. Unlike various existing fact-checking software, this software focuses on psychological communication style detection and labeling.

Functionality
MCDCAS is designed to provide a comprehensive solution to the challenges mentioned above. It is a software library that allows users to upload transcripts of conversations or individual messages. The software then analyzes the text, applying labels that indicate potential manipulative communication styles.

This tool offers two main use cases:

Large Language Model Evaluation: As more sophisticated models continue to emerge, the need for tools to measure their alignment with human values grows concurrently. MCDCAS enables developers and researchers to evaluate and benchmark language model outputs for potential manipulative expressions. This can help inform adjustments and improvements to these models, promoting transparency, safety, and ethical considerations in AI development.

Human Communication Analysis: The application of MCDCAS extends beyond AI. By analyzing human-to-human conversations, MCDCAS can help identify manipulative behaviors and patterns. This capability can provide critical support to individuals who may be victims of manipulation, raising awareness and facilitating the development of counter-strategies.

In the future, the plan is to expand MCDCAS’s reach by offering it as a Software as a Service (SaaS) solution. Users will be able to access its functionalities via a web-based JSON API and a user-friendly graphical interface.

The vision for MCDCAS is to foster a culture of communication that is more transparent, equitable, and free from manipulation. We believe that by illuminating the nuances of language, we can contribute to a better understanding between AI and humans, as well as among humans themselves.

Use Cases
Benchmarking of New LLM Models:
Benchmarking LLM resistance to manipulation from users. Even if the user input is manipulative, the LLM output should not be manipulative.
Benchmarking LLM outputs for the presence of manipulation in the case of benign user inputs.
Supporting Humans: Assisting in their communication with other humans as well as with LLMs.
Evaluation of News Articles and Blog Posts.
For Software Providers: Automatic detection of certain types of prompt injections from end users.
How It Works
The software is able to detect a detailed list of manipulative communication styles.
Locations where a particular manipulative style is detected are highlighted/annotated with location markers.
The same location may also get annotated with multiple markers if appropriate.
The software provides three main outputs:
Annotation of the input conversation with labels for mixed qualitative/quantitative analysis purposes.
Summary metrics for quantitative benchmark purposes, containing the total counts of occurrence of each manipulation style per conversation participant.
For purely qualitative analysis, a general descriptive summary text of the involved conversation participants.
Internally, the software uses two prompts:
A closed-ended prompt where the software or end user provides a concrete list of manipulation styles. The software asks the LLM to label the conversation using the labels in this list.
An open-ended prompt where the LLM can describe the communication styles present in the input conversation in an unrestricted way. The purpose is to extend the labels list used in the closed-ended prompt.
Some entries in the default labels list have partially or entirely overlapping meanings to account for the LLM's tendency to use different labels for similar or the same things. Unknown labels provided by the LLM end up in the output field unexpected_labels.
This software differs from lie detection/fact-checking software as it only focuses on communication style without relying on external knowledge bases (except for the use of a language model).
Additional Features
Input Text Anonymization: Person names, organization names, place names, etc., are replaced with abstract names like Person A, Person B, etc. This serves two purposes:
Anonymized input may make the LLM evaluations more fair.
Anonymized input significantly reduces the risk of private or sensitive data leakage.
Example Output
For a quick online preview, an automatic human-readable PDF rendering of the sample output is available here: data/test_evaluation.pdf

Sample JSON output can be found here: data/test_evaluation.json

Sample HTML output from which the above PDF was generated can be found here: data/test_evaluation.html

In addition to labeled highlights on the field expressions, there is summary statistics with total counts of manipulation styles for data analysis purposes in the field counts. A qualitative summary text is provided in the field qualitative_evaluation.

Usage
Windows setup:

vbnet
Copy code
set OPENAI_API_KEY=<your key here>
Linux setup:

javascript
Copy code
export OPENAI_API_KEY=<your key here>
Main command:

css
Copy code
python Recogniser.py ["input_file.txt" ["output_file.json" ["list_of_labels.txt" ["ignored_labels.txt"]]]]
The user-provided files are expected to be in the same folder as the main Python script unless an absolute path is provided. If run without arguments, sample files in the data folder are used. If the user provides an input file name but no output file name, the output JSON file name will be calculated as input filename + _evaluation.json, and the output HTML and PDF file names will be calculated as input filename + _evaluation.html and input filename + _evaluation.pdf.

Input Format Example
The input conversation is provided as a UTF-8 text file with a log of a conversation.

less
Copy code
Person A: Their message.
Person B: Response text.
Person A: More messages. And more sentences in that message.
Person B: The input continues as long as the conversation to be analyzed.
Etc...
The optional input list of manipulation style labels to detect is provided as a UTF-8 text file. The labels are separated by newlines. The data folder contains a list of default labels in the file default_labels.txt which is used when a user does not supply their own list of labels. The list format example follows.

diff
Copy code
- Diminishing
- Ignoring
- Victim playing
Etc...
See default_labels.txt for the complete list of default labels.

Output Format Example
The software produces output in four formats:

HTML: Contains rendering of the conversation with highlighted relevant parts and their labels.
PDF: A version of the above HTML for setting up PDF rendering support, see install_steps.txt. See the Example Output section in this document for links to examples.
JSON: Contains the following structure. The numbers after labels in expressions dictionaries indicate confidence level in percent. The confidence level is computed by sampling LLM labeling output multiple times according to the SampleCount parameter in the configuration file. Low confidence labels are automatically removed corresponding to the DefaultLabelThresholdSamplePercent parameter.
json
Copy code
{
  "error_code": 0,
  "error_msg": "",
  "sanitised_text": "Slightly modified and optionally anonymised input text",
  "expressions": [
    {
      "person": "Person B",
      "start_char": 9,
      "end_char": 29,
      "message_index": 0,
      "line_number": 1,
      "text": "Their message.",
      "labels": {
        "Ignoring": 100
      }
    },
    {
      "person": "Person B",
      "start_char": 109,
      "end_char": 282,
      "message_index": 2,
      "line_number": 5,
      "text": "More messages. And more sentences in that message.",
      "labels": {
        "Diminishing": 75,
        "Invalidation": 60
      }
    },
    ...
  ],
  "counts": {
    "Person B": {
      "Diminishing": 7,
      "Invalidation": 5,
      "Victim playing": 2,
      "Manipulation": 4,
      "Exaggeration and dramatization": 1,
      "Aggression": 2,
      "Changing the topic": 1,
      "Ignoring": 1
    },
    "Person A": {
      "Impatience": 1
    }
  },
  "unexpected_labels": [],  // contains a list of labels which were not requested, but were present in LLM output regardless
  "raw_expressions_labeling_response": "Response from LLM based on which the computer-readable parsed data above is calculated.",
  "qualitative_evaluation": "Another text from LLM providing a general descriptive summary of the participants involved."
}
Future Plans
Data Improvements
Creating a list of conversation data sources/databases. Possible sources:
Quora
Reddit
Potential data source recommendations from Esben Kran:
https://talkbank.org/
https://childes.talkbank.org/
https://docs.google.com/document/d/1boRn_hpVfaXBydc3C18PTsJVutOIsM3dF3sJyFjq-vc/edit
Create a gold standard set of labels of manipulation styles. One potential source of labels could be existing psychometric tests.
Create a gold standard set of conversations and messages potentially containing manipulative themes.
Create a gold standard set of evaluations for a set of prompts. This can be done by collecting labelings from expert human evaluators.
New Functionalities
Support for single-message labeling. Currently, the algorithm expects a conversation as input, but with trivial modifications, it could also be applied to single messages or articles given that they have sufficient length.
Returning logit scores over the conversation for each person and label. Example:
css
Copy code
"logits_summary": {
    "Person A": {
        "Invalidation": 0.9,
        "Victim playing": 0.7,
        "Exaggeration and dramatization": 0.2
    }
}
Handling of similar labels with overlapping semantic themes. This task may involve computing correlations between raw labels and rotating/computing independent traits from the raw labels.
Add support for open-source models available at HuggingFace.
Software Tuning
Improving error handling.
Invalid LLM output detection. Sometimes LLM produces results in a different format than expected.
New Related Apps
Building and setting up a web-based API endpoint.
Building and setting up a web-based user interface for non-programmer end users.
Publishing a Slack plugin.
Experiments
Test manipulation detection against various known prompt injection prompts.
Test manipulation detection against general prompt databases (for example, AutoGPT database).
Benchmark various known LLMs:
LLM resistance to manipulation from users. Even if the user input is manipulative, the LLM output should not be manipulative.
Measure the presence of manipulation in LLM outputs in case of benign user inputs.
Look for conversations on the theme of Waluigi Effect (https://www.lesswrong.com/posts/D7PumeYTDPfBTp3i7/the-waluigi-effect-mega-post).
