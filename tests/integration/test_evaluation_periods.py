"""Development defaults and historical-period labels use actual sessions."""
from types import SimpleNamespace

import pytest

from src.evaluation.periods import describe_test_period, warn_for_explored_test
from src.training.cli import build_parser
from src.training.config import resolve_training_config


def test_default_and_empty_yaml_both_use_validation(tmp_path):
    default = build_parser().parse_args([])
    resolve_training_config(default)
    assert default.evaluation_scope == 'validation'
    config = tmp_path/'training.yaml'
    config.write_text('{}')
    args = build_parser().parse_args(['--training-config', str(config)])
    resolve_training_config(args)
    assert args.evaluation_scope == 'validation'
    explicit = build_parser().parse_args(['--training-config', str(config), '--evaluation-scope', 'test'])
    resolve_training_config(explicit)
    assert explicit.evaluation_scope == 'test'


@pytest.mark.parametrize('dates', [
    ['2024-01-02', '2025-12-30'], ['2023-01-03','2024-01-02','2025-12-30'],
    ['2025-06-02','2025-06-03'],
])
def test_known_reused_sessions_are_exploratory_regardless_of_cli_dates(dates):
    period = describe_test_period(SimpleNamespace(dates=dates), 'test')
    assert period['status'] == 'reused_exploratory_period_not_independent_holdout'
    assert period['start'] == dates[0]
    with pytest.warns(RuntimeWarning, match=dates[0]):
        warn_for_explored_test(period)


def test_unknown_period_does_not_claim_independence():
    period = describe_test_period(SimpleNamespace(dates=['2026-01-02','2026-01-05']), 'test')
    assert period['status'] == 'custom_period_freshness_unverified'
    assert not period['independent_holdout_verified']
    # Only actual evaluated sessions count, not missing dates between endpoints.
    period = describe_test_period(SimpleNamespace(dates=['2023-01-03','2026-01-05']), 'test')
    assert not period['overlaps_explored_period']


def test_validation_has_no_test_dates_and_missing_test_tensor_is_rejected():
    assert describe_test_period(None, 'validation')['status'] == 'not_evaluated_validation_only'
    with pytest.raises(ValueError, match='prepared test tensor'):
        describe_test_period(None, 'test')
