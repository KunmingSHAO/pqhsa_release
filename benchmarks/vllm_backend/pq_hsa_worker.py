"""In-repo vLLM V1 GPU worker that installs PQ-HSA on every rank.

Pass ``worker_cls='benchmarks.vllm_backend.pq_hsa_worker.PQHSAWorker'`` to
``LLM()``. Does not edit vLLM site-packages. TP=1 in-process runs keep the default Worker.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

from vllm.v1.worker.gpu_worker import Worker

from benchmarks.vllm_backend.worker_hook import (
    dump_rank_stats,
    install_in_process,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import ModelRunnerOutput


class PQHSAWorker(Worker):
    def init_device(self):
        install_in_process(rank=self.rank)
        super().init_device()

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> Optional["ModelRunnerOutput"]:
        try:
            return super().execute_model(scheduler_output)
        finally:
            try:
                dump_rank_stats(self.rank, reason="execute_model")
            except Exception:
                pass

    def pq_hsa_set_enable(self, on: bool) -> str:
        os.environ["PQ_HSA_ENABLE"] = "1" if on else "0"
        dump_rank_stats(self.rank, reason="set_enable")
        return os.environ["PQ_HSA_ENABLE"]

    def pq_hsa_runtime_stats(self) -> dict:
        dump_rank_stats(self.rank, reason="rpc_stats")
        from benchmarks.vllm_backend.pq_hsa_decode_runtime import pq_hsa_runtime_stats

        return {
            "rank": int(self.rank),
            "pid": os.getpid(),
            "pq_hsa_enable": os.environ.get("PQ_HSA_ENABLE", "0"),
            **pq_hsa_runtime_stats(),
        }

    def pq_hsa_reset_stats(self) -> bool:
        from benchmarks.vllm_backend.pq_hsa_decode_runtime import (
            reset_pq_hsa_runtime_stats,
        )

        reset_pq_hsa_runtime_stats()
        dump_rank_stats(self.rank, reason="reset_stats")
        return True
