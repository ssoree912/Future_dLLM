"""ARC-Challenge on the *test* split, not the dev split OpenCompass defaults to.

Identical to opencompass/configs/datasets/ARC_c/ARC_c_gen_1e0de5.py in prompt,
retriever and scoring -- the only change is `path`, from `ai2_arc-dev` to
`ai2_arc-test`.

Why: `ARC_c_gen` reads ARC-Challenge-Dev.jsonl (299 rows), and ARCDataset drops
every question that does not have exactly four choices -- the template is fixed
at A/B/C/D and `first_option_postprocess` scores against 'ABCD' -- leaving 295.
The test split is 1,172 rows, 1,165 of them four-choice, so it is four times the
items and the split the ARC-C column is normally reported on.

The two are not interchangeable in one column: a 295-item dev number and a
1,165-item test number are different populations, so every row read against each
other has to come from the same one.
"""
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import ARCDataset
from opencompass.utils.text_postprocessors import first_option_postprocess

ARC_c_reader_cfg = dict(
    input_columns=['question', 'textA', 'textB', 'textC', 'textD'],
    output_column='answerKey')

ARC_c_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt=
                    'Question: {question}\nA. {textA}\nB. {textB}\nC. {textC}\nD. {textD}\nAnswer:'
                )
            ], ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

ARC_c_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'),
)

# `abbr` says which split produced the row, so a dev run and a test run never
# land in the same results file or the same summary column.
ARC_c_datasets = [
    dict(
        abbr='ARC-c-test',
        type=ARCDataset,
        path='opencompass/ai2_arc-test',
        name='ARC-Challenge',
        reader_cfg=ARC_c_reader_cfg,
        infer_cfg=ARC_c_infer_cfg,
        eval_cfg=ARC_c_eval_cfg,
    )
]
