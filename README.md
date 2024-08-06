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
