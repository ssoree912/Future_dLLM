"""공식 dataset2prompt / dataset2maxlen 에서 lm-eval task 파일을 생성한다."""
import json, pathlib, re

HERE = pathlib.Path(__file__).resolve().parent
CFG = HERE.parent.parent.parent / 'data' / 'longbench' / 'config'
OUT = HERE / 'longbench'
prompts = json.load(open(CFG / 'dataset2prompt.json'))
maxlen  = json.load(open(CFG / 'dataset2maxlen.json'))

# LongBench 공식 채점 함수 -> vendored metrics.py 의 이름
METRIC = {
    'narrativeqa': 'get_qa_f1_with_score', 'qasper': 'get_qa_f1_with_score',
    'multifieldqa_en': 'get_qa_f1_with_score', 'hotpotqa': 'get_qa_f1_with_score',
    '2wikimqa': 'get_qa_f1_with_score', 'musique': 'get_qa_f1_with_score',
    'triviaqa': 'get_qa_f1_with_score',
    'gov_report': 'get_rouge_with_score', 'qmsum': 'get_rouge_with_score',
    'multi_news': 'get_rouge_with_score', 'samsum': 'get_rouge_with_score',
    'trec': 'get_classification_with_score',
    'passage_retrieval_en': 'get_retrieval_with_score',
    'passage_count': 'get_count_with_score',
    'lcc': 'get_code_sim_with_score', 'repobench-p': 'get_code_sim_with_score',
}
TAG = {'samsum': 'fewshot', 'trec': 'fewshot', 'triviaqa': 'fewshot',
       '2wikimqa': 'multi', 'hotpotqa': 'multi', 'musique': 'multi',
       'qasper': 'single', 'narrativeqa': 'single', 'multifieldqa_en': 'single',
       'gov_report': 'summarization', 'qmsum': 'summarization',
       'multi_news': 'summarization', 'lcc': 'code', 'repobench-p': 'code',
       'passage_retrieval_en': 'synthetic', 'passage_count': 'synthetic'}

metrics_src = (HERE / 'metrics.py').read_text()
for old in OUT.glob('*.yaml'):
    old.unlink()

written = []
for task, fn in METRIC.items():
    if task not in prompts or fn not in metrics_src:
        print('  건너뜀:', task, fn); continue
    # 공식 템플릿의 {context}/{input} 을 jinja 로. 다른 중괄호는 없다.
    tmpl = prompts[task]
    assert set(re.findall(r'\{(\w+)\}', tmpl)) <= {'context', 'input'}, task
    doc = tmpl.replace('{context}', '{{context}}').replace('{input}', '{{input}}')
    doc = json.dumps(doc)          # 개행/따옴표를 YAML 안전하게
    second = fn.replace('get_', '').replace('_with_score', '') + '_score'
    (OUT / f'{task}.yaml').write_text(f'''# Generated from LongBench's own config/dataset2prompt.json and
# config/dataset2maxlen.json, against the official zai-org/LongBench data.
tag:
  - longbench_{TAG[task]}_tasks
  - longbench_tasks
task: longbench_{task}
dataset_path: json
dataset_kwargs:
  data_files:
    test: LONGBENCH_DATA_DIR/{task}.jsonl
test_split: test
doc_to_text: {doc}
doc_to_target: '{{{{answers}}}}'
target_delimiter: ""
process_results: !function metrics.{fn}
generation_kwargs:
  max_gen_toks: {maxlen[task]}
  do_sample: false
  until: []
metric_list:
  - metric: "score"
    aggregation: mean
    higher_is_better: True
  - metric: "{second}"
    aggregation: mean
    higher_is_better: True
metadata:
  version: 6.0
''')
    written.append(task)
print('생성:', len(written), '->', ' '.join(sorted(written)))
