"""Day 1 — Automated validation pipeline.

Checks: missing values, type/format issues, range checks, schema conformance.
Inputs are pandas DataFrames (batches) or single records (dicts).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

import pandas as pd

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PHONE_RE = re.compile(r"^\+?\d{7,15}$")


@dataclass
class FieldRule:
    name: str
    required: bool = True
    dtype: str | None = None  # "str" | "int" | "float" | "bool" | "datetime"
    regex: re.Pattern | None = None
    min_value: float | None = None
    max_value: float | None = None
    allowed: Iterable[Any] | None = None


@dataclass
class ValidationIssue:
    field: str
    rule: str
    detail: str
    row_index: int | None = None


@dataclass
class ValidationReport:
    total_records: int
    passed: int
    failed: int
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return 0.0 if self.total_records == 0 else self.passed / self.total_records

    def to_dict(self) -> dict:
        return {
            "total_records": self.total_records,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": round(self.pass_rate, 4),
            "issues": [i.__dict__ for i in self.issues[:500]],
        }


# Default ruleset for KoraFlex transaction records
TRANSACTION_RULES: list[FieldRule] = [
    FieldRule("transaction_id", required=True, dtype="str"),
    FieldRule("user_id", required=True, dtype="str"),
    FieldRule("amount", required=True, dtype="float", min_value=0.0, max_value=10_000_000.0),
    FieldRule("currency", required=True, allowed={"NGN", "USD", "GHS", "KES"}),
    FieldRule("merchant_id", required=True, dtype="str"),
    FieldRule("timestamp", required=True, dtype="datetime"),
]

USER_RULES: list[FieldRule] = [
    FieldRule("user_id", required=True, dtype="str"),
    FieldRule("email", required=True, regex=EMAIL_RE),
    FieldRule("phone", required=True, regex=PHONE_RE),
    FieldRule("full_name", required=True, dtype="str"),
]


def _check_row(row: dict, rules: list[FieldRule], idx: int | None = None) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for r in rules:
        val = row.get(r.name)
        missing = val is None or (isinstance(val, float) and pd.isna(val)) or val == ""
        if missing:
            if r.required:
                issues.append(ValidationIssue(r.name, "required", "missing value", idx))
            continue
        if r.dtype == "float":
            try:
                v = float(val)
                if r.min_value is not None and v < r.min_value:
                    issues.append(ValidationIssue(r.name, "min_value", f"{v} < {r.min_value}", idx))
                if r.max_value is not None and v > r.max_value:
                    issues.append(ValidationIssue(r.name, "max_value", f"{v} > {r.max_value}", idx))
            except (TypeError, ValueError):
                issues.append(ValidationIssue(r.name, "dtype", f"expected float, got {val!r}", idx))
        elif r.dtype == "int":
            try:
                int(val)
            except (TypeError, ValueError):
                issues.append(ValidationIssue(r.name, "dtype", f"expected int, got {val!r}", idx))
        elif r.dtype == "datetime":
            try:
                pd.to_datetime(val)
            except Exception:
                issues.append(ValidationIssue(r.name, "dtype", f"bad datetime {val!r}", idx))
        if r.regex and not r.regex.match(str(val)):
            issues.append(ValidationIssue(r.name, "regex", f"regex mismatch {val!r}", idx))
        if r.allowed is not None and val not in r.allowed:
            issues.append(ValidationIssue(r.name, "allowed", f"{val!r} not in {sorted(r.allowed)}", idx))
    return issues


def validate_record(record: dict, rules: list[FieldRule]) -> ValidationReport:
    issues = _check_row(record, rules)
    return ValidationReport(total_records=1, passed=int(not issues), failed=int(bool(issues)), issues=issues)


def validate_dataframe(df: pd.DataFrame, rules: list[FieldRule]) -> ValidationReport:
    report = ValidationReport(total_records=len(df), passed=0, failed=0)
    for idx, row in df.iterrows():
        row_issues = _check_row(row.to_dict(), rules, idx=int(idx))
        if row_issues:
            report.failed += 1
            report.issues.extend(row_issues)
        else:
            report.passed += 1
    return report
