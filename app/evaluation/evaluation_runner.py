import json
import queue
import threading
from pathlib import Path

from evaluation.evaluator import Evaluator
from evaluation.llm_judge import LLMJudge
from evaluation.test_question import TestQuestion

NUM_RAG_WORKERS = 10
NUM_JUDGE_WORKERS = 10


class EvaluationRunner:
    DATASET_PATH = Path(__file__).parent / "golden_dataset.jsonl"

    def __init__(self, retriever, agent_factory, llm_factory=None):
        self.retriever = retriever
        self.agent_factory = agent_factory
        self.evaluator = Evaluator()
        # llm_factory() returns a BaseChatModel; if omitted, judge stage is skipped.
        self.llm_factory = llm_factory

    def _load_questions(self) -> list[TestQuestion]:
        questions = []
        with open(self.DATASET_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    questions.append(TestQuestion.from_dict(json.loads(line)))
        return questions

    def run(self) -> list[TestQuestion]:
        questions = self._load_questions()
        total = len(questions)
        print(f"Loaded {total} questions — starting {NUM_RAG_WORKERS} RAG workers")

        rag_queue: queue.Queue = queue.Queue()
        judge_queue: queue.Queue = queue.Queue()
        completed_lock = threading.Lock()
        completed_count = 0
        judged_lock = threading.Lock()
        judged_count = 0
        # Track how many answers enter judging (set once RAG stage finishes).
        judge_total = 0

        # Stage 1 — RAG workers (consumers of rag_queue, producers to judge_queue)
        def rag_worker():
            nonlocal completed_count
            agent = self.agent_factory()
            while True:
                try:
                    tq: TestQuestion = rag_queue.get(block=True, timeout=1)
                except queue.Empty:
                    break

                generated_answer = None
                try:
                    docs = self.retriever.retrieve(tq.question)
                    retrieved_chunk_ids = [doc.metadata.get("chunk_id", "") for doc in docs]
                    generated_answer = agent.invoke(tq.question)
                    tq.generated_answer = generated_answer
                    tq.compute_metrics(
                        generated_answer=generated_answer,
                        retrieved_chunk_ids=retrieved_chunk_ids,
                        evaluator=self.evaluator,
                    )
                except Exception as e:
                    print(f"[ERROR] RAG failed on {tq.question[:60]!r}: {e}")
                finally:
                    # Produce to judge stage regardless of success; None answer skips judging.
                    judge_queue.put((tq, generated_answer))
                    rag_queue.task_done()

                with completed_lock:
                    completed_count += 1
                    print(f"[{completed_count}/{total}] RAG done: {tq.question[:60]!r}")

        # Stage 2 — Judge workers (consumers of judge_queue)
        def judge_worker(llm_judge: LLMJudge):
            nonlocal judged_count, judge_total
            while True:
                item = judge_queue.get()

                if item is None:
                    # Sentinel: signal this worker to stop.
                    judge_queue.task_done()
                    break

                tq, generated_answer = item
                if generated_answer is not None:
                    try:
                        tq.judge_result = llm_judge.judge(
                            question=tq.question,
                            golden_answer=tq.golden_answer,
                            generated_answer=generated_answer,
                        )
                        with judged_lock:
                            judged_count += 1
                            current = judged_count
                            pending = judge_total
                        status = f"{current}/{pending}" if pending else str(current)
                        print(f"[JUDGE {status}] Judged: {tq.question[:60]!r}")
                    except Exception as e:
                        with judged_lock:
                            judged_count += 1
                            current = judged_count
                            pending = judge_total
                        status = f"{current}/{pending}" if pending else str(current)
                        print(f"[JUDGE {status}][ERROR] {tq.question[:60]!r}: {e}")
                judge_queue.task_done()

        # Start judge workers before RAG so they can consume items as soon as they appear.
        judge_threads = []
        if self.llm_factory is not None:
            print(f"Starting {NUM_JUDGE_WORKERS} judge workers")
            for _ in range(NUM_JUDGE_WORKERS):
                llm_judge = LLMJudge(self.llm_factory())
                t = threading.Thread(target=judge_worker, args=(llm_judge,), daemon=True)
                judge_threads.append(t)
                t.start()

        # Enqueue all questions and start RAG workers.
        for tq in questions:
            rag_queue.put(tq)

        rag_threads = [
            threading.Thread(target=rag_worker, daemon=True)
            for _ in range(NUM_RAG_WORKERS)
        ]
        for t in rag_threads:
            t.start()

        # Wait for all RAG work to complete.
        rag_queue.join()
        for t in rag_threads:
            t.join()

        # Now we know the exact number of answers entering the judge stage.
        with judged_lock:
            judge_total = sum(1 for tq in questions if tq.metrics is not None)

        # Send one sentinel per judge worker to shut them down cleanly.
        if judge_threads:
            for _ in judge_threads:
                judge_queue.put(None)
            judge_queue.join()
            for t in judge_threads:
                t.join()

        self._save_results(questions)
        self._print_report(questions)
        return questions

    def _save_results(self, questions: list[TestQuestion]) -> None:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)

        metrics_path = results_dir / "metrics_results.jsonl"
        with open(metrics_path, "w") as f:
            for tq in questions:
                if tq.metrics is None:
                    continue
                f.write(json.dumps({
                    "question": tq.question,
                    "category": tq.category,
                    "generated_answer": tq.generated_answer,
                    "mrr": tq.metrics.mrr,
                    "ndcg": tq.metrics.ndcg,
                    "keyword_coverage": tq.metrics.keyword_coverage,
                }) + "\n")
        print(f"Metrics results saved to {metrics_path}")

        judge_path = results_dir / "judge_results.jsonl"
        with open(judge_path, "w") as f:
            for tq in questions:
                if tq.judge_result is None:
                    continue
                jr = tq.judge_result
                f.write(json.dumps({
                    "question": tq.question,
                    "category": tq.category,
                    "generated_answer": tq.generated_answer,
                    "accuracy": {"score": jr.accuracy.score, "reasoning": jr.accuracy.reasoning},
                    "relevance": {"score": jr.relevance.score, "reasoning": jr.relevance.reasoning},
                    "groundedness": {"score": jr.groundedness.score, "reasoning": jr.groundedness.reasoning},
                    "conciseness": {"score": jr.conciseness.score, "reasoning": jr.conciseness.reasoning},
                    "overall": {"score": jr.overall.score, "reasoning": jr.overall.reasoning},
                }) + "\n")
        print(f"Judge results saved to {judge_path}")

    def _print_report(self, questions: list[TestQuestion]) -> None:
        scored = [tq for tq in questions if tq.metrics is not None]
        if not scored:
            print("No results to report.")
            return

        overall_mrr = sum(tq.metrics.mrr for tq in scored) / len(scored)
        overall_ndcg = sum(tq.metrics.ndcg for tq in scored) / len(scored)
        overall_kw = sum(tq.metrics.keyword_coverage for tq in scored) / len(scored)

        print("=" * 50)
        print("Retrieval Metrics")
        print("=" * 50)
        print(f"  MRR:               {overall_mrr:.4f}")
        print(f"  nDCG:              {overall_ndcg:.4f}")
        print(f"  Keyword Coverage:  {overall_kw:.4f}")

        by_category: dict[str, list[float]] = {}
        for tq in scored:
            by_category.setdefault(tq.category, []).append(tq.metrics.mrr)

        print()
        print("MRR by Category")
        print("-" * 50)
        for category, mrr_scores in sorted(by_category.items()):
            avg = sum(mrr_scores) / len(mrr_scores)
            print(f"  {category:<28} {avg:.4f}  (n={len(mrr_scores)})")

        # LLM-as-a-Judge report
        judged = [tq for tq in questions if tq.judge_result is not None]
        if not judged:
            print("=" * 50)
            return

        def _avg(attr: str) -> float:
            return sum(getattr(tq.judge_result, attr).score for tq in judged) / len(judged)

        print()
        print("=" * 50)
        print("LLM-as-a-Judge Scores  (1–5 scale)")
        print("=" * 50)
        criteria = ["accuracy", "relevance", "groundedness", "conciseness", "overall"]
        for c in criteria:
            print(f"  {c.capitalize():<16} {_avg(c):.2f} / 5")

        # Spot-check: any answer scoring ≤ 2 on any criterion (excluding overall).
        low_scorers = [
            tq for tq in judged
            if any(
                getattr(tq.judge_result, c).score <= 2
                for c in ["accuracy", "relevance", "groundedness", "conciseness"]
            )
        ]
        if low_scorers:
            print()
            print(f"Low-score flag (≤2 on any criterion) — {len(low_scorers)} question(s):")
            print("-" * 50)
            for tq in low_scorers:
                jr = tq.judge_result
                row = {
                    c: getattr(jr, c).score
                    for c in ["accuracy", "relevance", "groundedness", "conciseness"]
                }
                scores_str = "  ".join(f"{c[0].upper()}:{s}" for c, s in row.items())
                print(f"  [{scores_str}]  {tq.question[:60]!r}")

        print("=" * 50)
