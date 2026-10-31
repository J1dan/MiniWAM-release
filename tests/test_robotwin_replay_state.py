import copy

import pytest

from scripts.eval.robotwin_replay_state import (
    CLUTTER_RULE, STACK_RULE, clutter_defaults, normalize_snapshot,
)


def case():
    return dict(task_name='dump_bin_bigbin', task_config='demo_randomized',
        scene_snapshot=dict(models={'cluttered_obj': {'old': True}, 'object': {'scale': 1}},
                            setup_info={'cluttered_table_info': []}, actors=[{'pose': [1,2,3]}]),
        scene_normalization=[CLUTTER_RULE])


def test_alias_only_removed_when_both_scenes_have_no_clutter():
    c = case()
    normalized = normalize_snapshot(c, c['scene_snapshot'])
    assert 'cluttered_obj' not in normalized['models']
    assert normalized['actors'] == c['scene_snapshot']['actors']
    assert normalized['models']['object'] == {'scale': 1}
    assert 'cluttered_obj' in c['scene_snapshot']['models']
    actual = copy.deepcopy(c['scene_snapshot'])
    actual['setup_info']['cluttered_table_info'] = [{'object_type': 'bottle'}]
    with pytest.raises(ValueError, match='cluttered scene'):
        normalize_snapshot(c, actual)


def test_unknown_normalization_and_wrong_task_fail_closed():
    c = case()
    for rule in ('drop_actors', STACK_RULE):
        c['scene_normalization'] = [rule]
        with pytest.raises(ValueError, match='Unknown or inapplicable'):
            normalize_snapshot(c, c['scene_snapshot'])


def test_mutable_bounds_reproduce_upstream_addition_without_cross_case_drift():
    c = case()
    c['replay_setup_state'] = {'clutter_bounds_prior_calls': 3}
    defaults = clutter_defaults(c)
    expected = [-0.59, 0.59]
    for _ in range(3):
        for i in range(2):
            expected[i] += 0.3
    assert defaults[1] == expected
    defaults[1][0] += 100
    assert clutter_defaults(c)[1] == expected
    c['task_name'] = 'adjust_bottle'
    with pytest.raises(ValueError, match='inapplicable'):
        clutter_defaults(c)
