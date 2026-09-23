"""Compare frozen and adapted training statements without constructing the model."""

import ast
from contextlib import nullcontext
import os
from pathlib import Path
from types import SimpleNamespace

from cross_dataset.datasets.common import PROJECT_ROOT
from cross_dataset.tests.cpu_only import CPUOnlyTest


def main_tree(path):
    return next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "main")


class TrainingCompatibilityTests(CPUOnlyTest):
    def test_training_loop_has_only_two_module_access_adaptations(self):
        original = main_tree(PROJECT_ROOT / "train.py")
        adapted = main_tree(PROJECT_ROOT / "cross_dataset/train.py")
        original_loop = next(n for n in original.body if isinstance(n, ast.While))
        adapted_loop = next(n for n in adapted.body if isinstance(n, ast.While))
        source = ast.unparse(adapted_loop).replace("getattr(my_model, 'module', my_model)", "my_model.module")
        normalized = ast.parse(source).body[0]
        self.assertEqual(ast.dump(original_loop), ast.dump(normalized))
        # Accelerator parameters, optimizer call, LR construction and prepare order.
        for name in ("accelerator", "optimizers", "optimizer", "warm_up", "scheduler",
                     "train_dataloader", "val_dataloader"):
            def assignments(tree):
                return [ast.dump(n) for n in tree.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
            self.assertEqual(assignments(original), assignments(adapted), name)
        prepare = lambda tree: [ast.dump(n) for n in tree.body if isinstance(n, ast.Assign)
                               and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                               and n.value.func.attr == "prepare"]
        self.assertEqual(prepare(original), prepare(adapted))

    def test_lightweight_training_call_trace_and_original_iteration_labels(self):
        old = main_tree(PROJECT_ROOT / "train.py")
        new = main_tree(PROJECT_ROOT / "cross_dataset/train.py")
        for wrapped in (False, True):
            for accumulation in (1, 2):
                expected = self.loop_trace(old, wrapped=True, accumulation=accumulation)
                actual = self.loop_trace(new, wrapped=wrapped, accumulation=accumulation)
                self.assertEqual(actual, expected)
                forward_iters = [event[1] for event in actual if event[0] == "forward"]
                self.assertEqual(forward_iters, list(range(6)))
                # max_train_steps=2 controls iter_end, not the 2 epochs x 3 batches stop.
                self.assertTrue(all(e[2] == 2 for e in actual if e[0] == "forward"))
                if accumulation == 1:
                    self.assertEqual([e[1] for e in actual if e[0] == "save"], [0, 4])
                    self.assertEqual([e[1] for e in actual if e[0] == "validation"], [0, 2, 4])

    def loop_trace(self, tree, wrapped, accumulation):
        trace, state = [], {"iteration": 0}
        record = lambda name, *values: trace.append((name, *values))

        class Model:
            def train(self): record("train")
            def eval(self): record("eval")
            def parameters(self): return ()
            def forward(self, batch, split, iter, iter_end):
                state["iteration"] = iter
                record("forward", iter, iter_end)
                return (SimpleNamespace(item=lambda: 0), {}, *([None] * 7))
            def validation_step(self, batch, path):
                record("validation", state["iteration"])
                return {}

        model = Model()
        if wrapped:
            model.module = model

        class Accelerator:
            is_main_process = True
            @property
            def sync_gradients(self): return (state["iteration"] + 1) % accumulation == 0
            def accumulate(self, model): return nullcontext()
            def backward(self, loss): record("backward")
            def clip_grad_norm_(self, params, maximum):
                record("clip", maximum)
                return 0
            def wait_for_everyone(self): record("wait")
            def save_state(self, path): record("save", state["iteration"])
            def log(self, log, step): record("log", step)

        cfg = SimpleNamespace(max_train_steps=2, grad_max_norm=1, save_freq=4, val_freq=2)
        namespace = dict(epoch=0, max_num_epochs=2, my_model=model, time=SimpleNamespace(time=lambda: 0),
                         train_dataloader=[{}, {}, {}], val_dataloader=[{}], accelerator=Accelerator(),
                         optimizer=SimpleNamespace(zero_grad=lambda: record("zero_grad"),
                                                   step=lambda: record("optimizer_step"), param_groups=[{"lr": 1e-4}]),
                         scheduler=SimpleNamespace(step=lambda: record("scheduler_step")), cfg=cfg, global_iter=0,
                         os=os, osp=os.path, args=SimpleNamespace(work_dir="/tmp/target-training-trace"),
                         mmengine=SimpleNamespace(utils=SimpleNamespace(symlink=lambda *args: record("symlink"))),
                         logger=None, print_freq=100)
        loop = next(n for n in tree.body if isinstance(n, ast.While))
        exec(compile(ast.Module(body=[loop], type_ignores=[]), "training-loop", "exec"), namespace)
        return trace
