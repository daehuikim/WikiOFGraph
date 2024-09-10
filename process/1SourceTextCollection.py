from datasets import load_dataset
import re
import random
import argparse

class SentenceProcessor:
    def __init__(self, excluded_start_words, intended_number):
        self.excluded_start_words = excluded_start_words
        self.intended_number = intended_number
        self.source_text_candidates = []

    def remove_parentheses(self, text):
        while re.search(r'\([^()]*\)', text):
            text = re.sub(r'\([^()]*\)', '', text)
        text = re.sub(r'[()]', '', text)
        text = ' '.join(text.split())
        return text

    def sentence_split(self, text):
        return re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)

    def process_dataset(self, dataset_name, dataset_version):
        dataset = load_dataset(dataset_name, dataset_version)
        selected = random.sample(range(0, len(dataset['train'])), self.intended_number)

        for i in selected:
            sentences = self.sentence_split(dataset['train'][i]['text'])
            first_sentence = sentences[0]
            if 10 <= len(first_sentence) <= 500 and not any(first_sentence.startswith(word) for word in self.excluded_start_words):
                self.source_text_candidates.append(self.remove_parentheses(first_sentence))

    def save_texts(self, file_path):
        with open(file_path, 'w') as f:
            for item in self.source_text_candidates:
                f.write("%s\n" % item)

def main():
    parser = argparse.ArgumentParser(description="Process some sentences from a dataset.")
    parser.add_argument('--save_path', type=str, required=True, help='The path to save the annotations')
    parser.add_argument('--intended_number', type=int, required=True, help='The number of sentences to process')

    args = parser.parse_args()

    excluded_start_words = ["is", "was", "are", "were", "he", "she", "it", "they", "we", ","]
    intended_number = args.intended_number

    processor = SentenceProcessor(excluded_start_words, intended_number)
    processor.process_dataset("wikipedia", "20220301.en")
    processor.save_texts(args.save_path)

if __name__ == "__main__":
    main()
