from os.path import join

from gpv2.data.dataset import Task
from gpv2.data.dce_dataset import DceDataset
from gpv2.eval.evaluation import get_evaluator
from gpv2.model.model import GPVExampleOutput
from gpv2.train.evaluator import DceClsEvaluator, DceVqaEvaluator
from gpv2.utils.py_utils import load_json_object

SRC = "/Users/chris/Desktop/dce-baselines"

TO_EVAL = [
  "swin_classification_dce_test.json",
  "swin_classification_dce_val.json",
  # "vilt_vqa_dce_test.json",
  # "vilt_vqa_dce_val.json"
]


def main():
  for file in TO_EVAL:
    print(file)
    parts = file.split(".")[0].split("_")
    task = parts[1]
    if task == "classification":
      task = Task.CLS
    else:
      task = Task(task)
    against = DceDataset(task, parts[-1])
    examples = against.load()

    if task == Task.VQA:
      qid_to_gpv_id = {int(ex.meta["qid"]): ex.gpv_id for ex in examples}
    else:
      qid_to_gpv_id = {int(ex.gpv_id.split("-")[-1]): ex.gpv_id for ex in examples}

    data = load_json_object(join(SRC, file))
    gpv_data = {}
    for ans in data:
      gpv_data[qid_to_gpv_id[ans["id"]]] = GPVExampleOutput(None, None, [ans["answer"]], [1.0])

    evaluator = get_evaluator(against)[0]
    if isinstance(evaluator, DceClsEvaluator):
      evaluator.top_k = None
    elif isinstance(evaluator, DceVqaEvaluator):
      evaluator.top_k = None
    print(evaluator.evaluate(examples, gpv_data))


if __name__ == '__main__':
  main()