# WikiOFGraph
We provide actual codes for reproducing our dataset.

Generated toy examples are provided in [intermediate](./intermediate/).

## 1 .Source text collection
run ```python 1SourceTextcollection.py --intended_number 10 --save_path intermediate/sourcetext/example_text.txt```

## 2. Graph Extraction
run ```python 2GraphExtraction.py --input_data_path intermediate/sourcetext/example_text.txt --intended_number 10 --save_path intermediate/extracted/example_extracted.txt```
If you want to adjust other parameters you can give much argument in your running script.

## 3. Data-QuestEval Filtering
run ```python 3DataQuestEvalFiltering.py --source_text_path intermediate/sourcetext/example_text.txt --extracted_graph_path intermediate/extracted/example_extracted.txt --save_path intermediate/filtered/example_filtered.txt```