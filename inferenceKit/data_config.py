from functools import partial
from .dataset import *
import os
BENCHMARK_PATH= os.environ.get('BENCHMARK_ROOT',r'./benchmark') # r"./benchmark"
supported_dataset = {
    'math': partial(MathDataset, dataset_name='math', dataset_path=os.path.join(BENCHMARK_PATH,"math_test500_dataset.json")),
    'gsm8k': partial(GSM8KDataset, dataset_name='gsm8k', dataset_path=os.path.join(BENCHMARK_PATH,"gsm8k_test1319_dataset.json")),
    'gsm8ktoy': partial(GSM8KDataset, dataset_name='gsm8ktoy', dataset_path=os.path.join(BENCHMARK_PATH,"gsm8k_toy20_dataset.json")),
    'mathtoy': partial(MathDataset, dataset_name='mathtoy', dataset_path=os.path.join(BENCHMARK_PATH,"math_toy20_dataset.json")),
    'math100': partial(MathDataset, dataset_name='math100', dataset_path=os.path.join(BENCHMARK_PATH,"math_100_dataset.json")),
    'gsm8k100': partial(GSM8KDataset, dataset_name='gsm8k100', dataset_path=os.path.join(BENCHMARK_PATH,"gsm8k_100_dataset.json")),
    'gsm8k500': partial(GSM8KDataset, dataset_name='gsm8k500', dataset_path=os.path.join(BENCHMARK_PATH,"gsm8k_500_dataset.json"))
}