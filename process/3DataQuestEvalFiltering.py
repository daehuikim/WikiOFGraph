import json
import random
from questeval.questeval_metric import QuestEval

class DataQuestEvalFilter:
    def __init__(self, questeval_task, source_text_path, extracted_graph_path, save_path):
        self.questeval = QuestEval(task=questeval_task)
        self.source_text_path = source_text_path
        self.extracted_graph_path = extracted_graph_path
        self.save_path = save_path

    def parse_string_to_list(self, string):
        tuples = string.strip('[]').split('], [')
        result = []
        for tpl in tuples:
            values = tpl.split('|')
            result.append([values[0].strip('<S> ').strip(), values[1].strip('<P> ').strip(), values[2].strip('<O> ').strip()])
        return result

    def valid_f1(self, annotation, table):
        score = self.questeval.corpus_questeval(
            hypothesis=[annotation],
            sources=[table]
        )
        return score['corpus_score'] >= 0.3

    def list2rdf(self, triplet):
        triples = [f"(<S> {items[0]}| <P> {items[1]}| <O> {items[2]})" for items in triplet]
        random.shuffle(triples)
        return ', '.join(triples)

    def load_data(self):
        with open(self.source_text_path, 'r') as f:
            self.text_lines = f.readlines()
        
        with open(self.extracted_graph_path, 'r') as f:
            self.lines = f.readlines()

    def extract_results(self):
        self.results = []
        for i, line in enumerate(self.lines):
            if "Filtering result is" in line and i + 1 < len(self.lines):
                self.results.append(self.lines[i + 1].strip())

    def filter_results(self):
        self.result_filtered = []
        self.triplet_for_result = []
        for i, table in enumerate(self.results):
            table = table.replace("(", "[").replace(")", "]")
            try:
                item_list = self.parse_string_to_list(table)
                str_item_list = [str(' | '.join(i)) for i in item_list]
                self.result_filtered.append(self.text_lines[i])
                self.triplet_for_result.append(str_item_list)
            except Exception:
                continue

    def process_results(self):
        self.final_results = [
            (text, self.list2rdf(triplets))
            for text, triplets in zip(self.result_filtered, self.triplet_for_result)
            if self.valid_f1(text, triplets)
        ]

    def save_results(self):
        with open(self.save_path, 'w') as f:
            for result in self.final_results:
                json.dump({"triplet": result[1], "text": result[0].strip()}, f, ensure_ascii=False)
                f.write('\n')

def main():
    source_text_path = "YOUR_SOURCE_TEXT_PATH"
    extracted_graph_path = "YOUR_EXTRACTED_GRAPH_PATH"
    save_path = "YOUR_SAVE_PATH"

    evaluator = DataQuestEvalFilter(
        questeval_task="data2text",
        source_text_path=source_text_path,
        extracted_graph_path=extracted_graph_path,
        save_path=save_path
    )

    evaluator.load_data()
    evaluator.extract_results()
    evaluator.filter_results()
    evaluator.process_results()
    evaluator.save_results()

if __name__ == '__main__':
    main()
