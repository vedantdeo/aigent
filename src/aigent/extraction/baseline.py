"""Project 1a's classical baseline: TF-IDF and logistic regression on `direction` alone.

uv run python -m aigent.extraction.baseline

Graded by the harness that graded the LLMs, so its rows sit in the same table. Free and offline.
"""

from __future__ import annotations

import sys
import warnings
from collections import Counter
from collections.abc import Sequence
from typing import cast

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline

from aigent.config import (
    BASELINE_CS,
    BASELINE_CV_FOLDS,
    BASELINE_MAX_ITER,
    BASELINE_NGRAMS,
    BASELINE_TOP_TERMS,
    SYNTHETIC_SEED,
    SYNTHETIC_VALID_FRACTION,
)
from aigent.evals.dataset import Case, digest, load_jsonl
from aigent.evals.grade import Outcome, field_match
from aigent.evals.report import write_report
from aigent.evals.runner import Task, run_eval
from aigent.extraction.headlines import DATASET
from aigent.extraction.synthetic import generate, split

MODEL = "tfidf+logreg"
FIELD = "direction"


def fit(headlines: Sequence[str], labels: Sequence[str]) -> Pipeline:
    """TF-IDF over word n-grams into a logistic regression whose C is cross-validated on the
    training rows alone."""
    logreg = LogisticRegressionCV(cv=BASELINE_CV_FOLDS, max_iter=BASELINE_MAX_ITER)
    # set_params, because pyright types these from their defaults: Cs=10 reads as int.
    logreg.set_params(
        Cs=list(BASELINE_CS), scoring="accuracy", l1_ratios=(0.0,), use_legacy_attributes=False
    )
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=BASELINE_NGRAMS, sublinear_tf=True)),
            ("logreg", logreg),
        ]
    )
    with warnings.catch_warnings():
        # `flat` has three real examples, fewer than the folds; stratifying does what it can.
        warnings.filterwarnings("ignore", message="The least populated class")
        pipeline.fit(list(headlines), list(labels))
    return pipeline


def predict(pipeline: Pipeline, headline: str) -> str:
    return str(pipeline.predict([headline])[0])


def headline_of(case: Case) -> str | None:
    headline = case.input.get("headline")
    return headline if isinstance(headline, str) else None


def label_of(case: Case) -> str:
    return str(case.expected[FIELD])


def synthetic_training() -> tuple[list[str], list[str]]:
    """The toy fine-tune's own training split, so the two are taught the same headlines."""
    train, _ = split(generate(), SYNTHETIC_VALID_FRACTION, SYNTHETIC_SEED)
    return [row.headline for row in train], [row.label.direction for row in train]


def _outcome(case: Case, answer: str | None) -> Outcome:
    if answer is None:
        return Outcome(error=f"case {case.id} has no headline")
    return Outcome(output={FIELD: answer}, model=MODEL)


def trained_on(headlines: Sequence[str], labels: Sequence[str]) -> Task:
    """One model fitted up front, then asked about every case."""
    pipeline = fit(headlines, labels)

    def task(case: Case) -> Outcome:
        headline = headline_of(case)
        return _outcome(case, None if headline is None else predict(pipeline, headline))

    return task


def leave_one_out(cases: Sequence[Case]) -> Task:
    """Each case answered by a model fitted on every other case."""

    def task(case: Case) -> Outcome:
        headline = headline_of(case)
        if headline is None:
            return _outcome(case, None)
        rest = [c for c in cases if c.id != case.id]
        pipeline = fit([headline_of(c) or "" for c in rest], [label_of(c) for c in rest])
        return _outcome(case, predict(pipeline, headline))

    return task


def majority(cases: Sequence[Case]) -> Task:
    """The floor: the commonest label among the other cases, whatever the headline says."""

    def task(case: Case) -> Outcome:
        rest = Counter(label_of(c) for c in cases if c.id != case.id)
        return _outcome(case, rest.most_common(1)[0][0])

    return task


def top_terms(pipeline: Pipeline, k: int = BASELINE_TOP_TERMS) -> dict[str, list[str]]:
    """The `k` n-grams pulling hardest towards each label."""
    vectorizer = cast(TfidfVectorizer, pipeline.named_steps["tfidf"])
    logreg = cast(LogisticRegressionCV, pipeline.named_steps["logreg"])
    vocabulary = vectorizer.get_feature_names_out()
    weights = np.asarray(logreg.coef_)
    classes = [str(label) for label in np.asarray(logreg.classes_)]
    if len(classes) == 2:  # a binary fit keeps one row, for the second class
        weights = np.vstack([-weights[0], weights[0]])
    return {
        label: [str(vocabulary[i]) for i in np.argsort(row)[::-1][:k]]
        for label, row in zip(classes, weights, strict=True)
    }


def main(argv: list[str] | None = None) -> None:
    del argv
    cases = [case for case in load_jsonl(DATASET) if FIELD in case.expected]
    headlines, labels = synthetic_training()
    print(f"{DATASET.name}: {len(cases)} labelled cases; {len(headlines)} template headlines")

    run = run_eval(
        cases,
        {
            "majority": majority(cases),
            "synthetic": trained_on(headlines, labels),
            "leave_one_out": leave_one_out(cases),
        },
        {"direction": field_match((FIELD,))},
        dataset=DATASET.name,
        digest=digest(DATASET),
        model=MODEL,
        progress=False,
    )
    for variant in run.variants:
        rows = run.rows_for(variant)
        print(f"  {variant}: {sum(row.passed for row in rows)}/{len(rows)}")
    print("\nterms the template-trained model leans on:")
    for label, terms in top_terms(fit(headlines, labels)).items():
        print(f"  {label}: {', '.join(terms)}")
    print(f"\nreport: {write_report(run)}")


if __name__ == "__main__":
    main(sys.argv[1:])
