# NLP Final Project
# Multilingual-Video-Synopsys-generator
The Multilingual Video Synopsis Generator entails downloading the most recent video, utilising Google Speech Recognition to transcribe the audio material, tokenization, sentiment analysis, and translation into Arabic and French as preparation steps. Next, it uses pre-trained transformer models (BART and MBART) to generate summaries in English, Arabic, and French. The quality of these summaries is then assessed using ROUGE ratings. All in all, the project automates the summary process, allowing users to effectively extract important information from multilingual videos.

# Usage
After importing and installing all the dependencies for the project, start running the code.
1. Download youtube video using url and stave it in your local.
2. Fetch the downloaded video from local and then convert into audio(.wav) and then transcribe it into text.
3. preprocess the generated transcribed text using NLP techniques.
4. Translate the preprocessed text to arabic and french.
5. Feed that translated text to summarization transformer models(BART and MBART).
6. Evaluate the results(summary) via ROUGE score and Human evaluation.
