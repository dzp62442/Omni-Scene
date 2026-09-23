"""CPU/Gloo integration fixture. Launch with torch.distributed.run, never the model.

Example: python -m torch.distributed.run --standalone --nproc_per_node=2
         --module cross_dataset.tests.distributed_evaluation_cpu
"""

import os
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ACCELERATE_USE_CPU"] = "true"
# Accelerate 1.7 / torch 2.1 compatibility, scoped to this CPU fixture.
os.environ["ACCELERATE_TORCH_DEVICE"] = "cpu"

import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from torch.utils.data import DataLoader

from cross_dataset.evaluation import EvaluationShard, summarize
from cross_dataset.tests.test_evaluation_cpu import SyntheticDataset, synthetic_records


def main():
    with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("GPU forbidden")):
        accelerator = Accelerator(cpu=True)
        for count, maximum in ((5, None), (1, 1)):
            dataset = EvaluationShard(SyntheticDataset(), count, 2, accelerator.process_index, accelerator.num_processes)
            records = []
            for batch in DataLoader(dataset, batch_size=2):
                records.extend(synthetic_records(batch))
            gathered = gather_object(records)
            summary, ordered = summarize(gathered, [f"bin_{i}" for i in range(count)], 5, maximum)
            assert summary["evaluated_count"] == count
            assert summary["complete_split"] == (maximum is None)
            assert len({r["bin_token"] for r in ordered}) == count
            assert summary["padding_records_discarded"] == len(gathered) - count
            accelerator.print({"cpu_world_size": accelerator.num_processes, **summary})
        errors = gather_object(["synthetic rank failure"] if accelerator.process_index == 1 else [])
        assert errors == ["synthetic rank failure"]
        assert not torch.cuda.is_initialized()
        accelerator.end_training()


if __name__ == "__main__":
    main()
