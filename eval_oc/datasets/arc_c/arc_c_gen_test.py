"""ARC-Challenge on the test split, everything else vanilla OpenCompass.

OpenCompass's own ``ARC_c_gen`` points at ``opencompass/ai2_arc-dev`` -- the
299-item dev split. The other two sets in this suite are already complete
(GPQA diamond 198, PIQA validation 1838), so ARC-C is the one that needs
switching to test (1172 items, the split lm-eval's ``arc_challenge`` uses and
the one every published ARC-C number refers to).

Only ``path`` differs from ``ARC_c_gen_1e0de5``; prompt, retriever, inferencer,
evaluator and postprocessor are copied verbatim so the two splits are otherwise
scored identically.

Note ``ARCDataset.load`` drops any item that does not have exactly four
choices, so the effective count is slightly under 1172.
"""
from opencompass.datasets import ARCDataset
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
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
