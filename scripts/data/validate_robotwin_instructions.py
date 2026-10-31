"""Validate captured strings against the official templates and exact case bindings."""
import importlib.util
import json
from pathlib import Path
import re


def validate_instructions(cases, root):
    root = Path(root)
    spec = importlib.util.spec_from_file_location('official_instructions', root/'description/utils/generate_episode_instructions.py')
    official = importlib.util.module_from_spec(spec); spec.loader.exec_module(official)
    for case in cases:
        params = case['scene_snapshot']['episode_info']['info']
        templates = official.load_task_instructions(case['task_name'])[case['instruction_type']]
        templates = official.filter_instructions(templates, params)
        domains = {}
        for raw, value in params.items():
            key = raw.strip('{}')
            path = root/'description/objects_description'/f'{value}.json'
            if path.is_file():
                data = json.loads(path.read_text())
                descriptions = data.get(case['instruction_type'], [])
                if case['instruction_type'] == 'unseen' and not descriptions:
                    descriptions = data.get('seen', [])
                domains[key] = ['the '+s for s in descriptions]
            elif '/' in value or '\\' in value:
                raise ValueError(f'Missing official object description: {value}')
            elif len(key) == 1 and key.islower():
                domains[key] = ['the '+value+' arm']
            else:
                domains[key] = [value]
        valid = False
        for template in templates:
            pieces, position, names = [], 0, {}
            for match in re.finditer(r'\{([^}]+)\}', template):
                pieces.append(re.escape(template[position:match.start()]))
                key = match[1]
                if key in names:
                    pieces.append('(?P='+names[key]+')')
                else:
                    names[key] = f'v{len(names)}'
                    if not domains.get(key):
                        raise ValueError(f'Empty official binding domain: {key}')
                    pieces.append('(?P<'+names[key]+'>'+ '|'.join(re.escape(v) for v in domains[key])+')')
                position = match.end()
            pieces.append(re.escape(template[position:]))
            if re.fullmatch(''.join(pieces), case['instruction']):
                valid = True
                break
        if not valid:
            raise ValueError(f'Instruction does not describe captured case: {case["task_name"]}/{case["task_config"]}/{case["accepted_seed"]}')
    return dict(total_cases=len(cases), valid_instructions=len(cases), method='official_template_and_exact_case_binding_membership')
