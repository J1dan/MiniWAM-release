"""Explicit replay-only state; never changes stock expert acceptance or physics.

The historical fingerprint includes Python wrappers that can outlive a scene.
Only documented, provably inactive aliases may be removed. Physical fields stay
bit-exact. Mutable clutter bounds are restored *before* stock setup and scoped
to one case so resume order cannot change the reconstructed scene.
"""
import copy

STACK_TASKS = {'stack_blocks_two', 'stack_blocks_three'}
BIASED_TASKS = {'dump_bin_bigbin', 'put_bottles_dustbin'}
STACK_RULE = 'drop_models.last_actor'
CLUTTER_RULE = 'drop_models.cluttered_obj_when_no_clutter'


def normalization_rules(case):
    rules = []
    if case['task_name'] in STACK_TASKS:
        rules.append(STACK_RULE)
    if case['scene_snapshot'].get('setup_info', {}).get('cluttered_table_info') == []:
        rules.append(CLUTTER_RULE)
    return rules


def normalize_snapshot(case, snapshot):
    rules = case.get('scene_normalization', [])
    if not isinstance(rules, list) or len(rules) != len(set(rules)):
        raise ValueError('Invalid scene normalization rules')
    result = copy.deepcopy(snapshot)
    for rule in rules:
        if rule == STACK_RULE and case['task_name'] in STACK_TASKS:
            result.get('models', {}).pop('last_actor', None)
        elif rule == CLUTTER_RULE:
            # Both sides must independently assert that no clutter was placed.
            if any(s.get('setup_info', {}).get('cluttered_table_info') != []
                   for s in (case['scene_snapshot'], snapshot)):
                raise ValueError('Cannot normalize clutter metadata in a cluttered scene')
            result.get('models', {}).pop('cluttered_obj', None)
        else:
            raise ValueError(f'Unknown or inapplicable scene normalization: {rule}')
    return result


def clutter_defaults(case):
    state = case.get('replay_setup_state')
    if state is None:
        return None
    if (case['task_name'] not in BIASED_TASKS or
            case['task_config'] != 'demo_randomized' or
            set(state) != {'clutter_bounds_prior_calls'}):
        raise ValueError('Unknown or inapplicable replay setup state')
    count = state['clutter_bounds_prior_calls']
    if type(count) is not int or not 0 <= count <= 10000:
        raise ValueError('Invalid clutter setup call count')
    xlim = [-0.59, 0.59]
    # Repeated additions reproduce upstream floating-point arithmetic exactly.
    for _ in range(count):
        xlim[0] += 0.3
        xlim[1] += 0.3
    return (10, xlim, [-0.34, 0.34], [0.741])
