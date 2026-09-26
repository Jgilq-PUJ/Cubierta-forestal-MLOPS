"""DAG integrity: every DAG file must load through the Airflow DagBag.

Catches wiring errors the SQL helpers cannot see, such as a task argument
named after a reserved context key (``run_id``), missing imports or a DAG
registered under the wrong file.

Run inside the Airflow image:

    python tests/test_dag_integrity.py
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

DAGS_FOLDER = Path(__file__).resolve().parents[1] / "dags"
# The scheduler puts the dags folder on sys.path; a standalone DagBag does not.
sys.path.insert(0, str(DAGS_FOLDER))
EXPECTED = {
    "covertype_ingestion": "covertype_ingestion.py",
    "covertype_training_dataset": "covertype_training_dataset.py",
}
RESERVED_CONTEXT_KEYS = {"run_id", "dag_run", "ti", "task_instance", "params", "ds", "ts"}


def test_dags_load_without_errors() -> None:
    from airflow.models import DagBag

    bag = DagBag(dag_folder=str(DAGS_FOLDER))
    assert not bag.import_errors, bag.import_errors
    for dag_id, filename in EXPECTED.items():
        assert dag_id in bag.dags, f"{dag_id} not found, got {sorted(bag.dags)}"
        assert Path(bag.dags[dag_id].fileloc).name == filename, bag.dags[dag_id].fileloc


def test_task_arguments_do_not_shadow_context_keys() -> None:
    from airflow.models import DagBag

    bag = DagBag(dag_folder=str(DAGS_FOLDER))
    for dag in bag.dags.values():
        for task in dag.tasks:
            callable_ = getattr(task, "python_callable", None)
            if callable_ is None:
                continue
            # Positional XCom args are bound by name at execute time, so map
            # op_args onto the signature the same way the SDK will.
            names = list(inspect.signature(callable_).parameters)
            bound = set(names[: len(getattr(task, "op_args", ()) or ())])
            bound |= set(getattr(task, "op_kwargs", None) or {})
            clash = RESERVED_CONTEXT_KEYS & bound
            assert not clash, f"{dag.dag_id}.{task.task_id} binds reserved key(s) {clash}"


def main() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
