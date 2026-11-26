#!/usr/bin/env python3
"""Generate one SWE-bench prediction with an isolated agent and a vLLM endpoint."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from agent import solve
from sandbox import Sandbox, image_name


HERE = Path(__file__).resolve().parent
PUBLIC = ('instance_id','repo','base_commit','problem_statement')
ENABLE_THINKING = True
DEFAULTS = {'turns':40, 'context':32768, 'total_tokens':160000, 'max_output':2048,
            'temperature':0.6, 'command_timeout':90, 'stall':6}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def stable(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def protocol_hash(settings):
    return sha(stable([settings, {name: sha((HERE/name).read_bytes()) for name in ('run.py','agent.py','sandbox.py','image_configs.json')}]))


def write_result(path, value):
    with tempfile.NamedTemporaryFile('w', dir=path.parent, encoding='utf-8', delete=False) as tmp:
        json.dump(value, tmp, ensure_ascii=False, indent=2)
        tmp.write('\n')
    os.replace(tmp.name, path)


def required_task(path, instance_id):
    found = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                raw = json.loads(line)
                if raw['instance_id'] == instance_id:
                    found.append({k:raw[k] for k in PUBLIC})
    if len(found) != 1 or not all(isinstance(v,str) and v.strip() for v in found[0].values()):
        raise ValueError(f'Expected one complete public task for {instance_id}')
    return found[0]


class Vllm:
    def __init__(self, base_url, model, seed, temperature):
        self.base = base_url.rstrip('/')
        if not self.base.endswith('/v1'):
            raise ValueError('Pass an OpenAI-compatible base URL ending in /v1')
        self.model, self.seed, self.temperature = model, seed, temperature

    def call(self, url, payload):
        headers = {'Content-Type':'application/json'}
        if os.getenv('HFIT_API_KEY'):
            headers['Authorization'] = 'Bearer ' + os.environ['HFIT_API_KEY']
        req = Request(url, data=stable(payload), headers=headers)
        try:
            with urlopen(req, timeout=240) as response:
                return json.load(response)
        except HTTPError as exc:
            detail = exc.read(1536).decode(errors='replace')
            raise RuntimeError(f'HTTP {exc.code} from model service: {detail}') from exc

    def fields(self, messages):
        return {'model':self.model, 'messages':messages, 'chat_template_kwargs':{'enable_thinking':ENABLE_THINKING}}

    def tokens(self, messages):
        result = self.call(self.base[:-3] + '/tokenize', self.fields(messages))
        n = result.get('count')
        if type(n) is not int or n < 1:
            raise RuntimeError("vLLM /tokenize did not return a chat token count")
        return n

    def chat(self, messages, max_output, step):
        data = self.call(self.base + '/chat/completions', {
            **self.fields(messages), 'temperature':self.temperature,
            'top_p':0.95, 'top_k':20, 'seed':(self.seed + step - 1) % (2**31), 'max_tokens':max_output, 'stream':False})
        choice = data['choices'][0]
        message = choice['message']
        content = message.get('content') or ''
        if not isinstance(content, str):
            raise ValueError('Expected text completion')
        # vLLM exposes the analysis separately. Only the final answer goes to the agent or trace.
        has_reasoning = any(bool(message.get(field)) for field in ('reasoning', 'reasoning_content'))
        if '</think>' in content and (not has_reasoning or content.lstrip().startswith('<think>')):
            analysis, _, content = content.partition('</think>')
            has_reasoning = has_reasoning or bool(analysis.removeprefix('<think>').strip())
        elif content.lstrip().startswith('<think>'):
            has_reasoning = True
            content = ''
        usage = data['usage']
        for k in ('prompt_tokens', 'completion_tokens'):
            if type(usage.get(k)) is not int or usage[k] < 0:
                raise ValueError('Missing model usage accounting: ' + k)
        accounting = {k:usage[k] for k in ('prompt_tokens','completion_tokens')}
        accounting['has_reasoning'] = has_reasoning
        details = usage.get('completion_tokens_details') or {}
        if details.get('reasoning_tokens') is not None:
            count = details['reasoning_tokens']
            if type(count) is not int or not 0 <= count <= usage['completion_tokens']:
                raise ValueError('Invalid reasoning token accounting')
            accounting['reasoning_tokens'] = count  # Already included in completion_tokens.
        return content, accounting, choice['finish_reason']


def safe_component(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', value):
        raise ValueError('Use a simple served-model alias or instance ID as path component')
    return value


def run_slot(args, *, env_class=Sandbox, model_class=Vllm):
    model, instance_id = safe_component(args.model), safe_component(args.instance_id)
    if not re.fullmatch('[01]{5}', args.config) or args.seed < 0:
        raise ValueError('Expected five configuration bits and a nonnegative seed')
    settings = {k:getattr(args, k) for k in DEFAULTS}
    if any(v <= 0 for k, v in settings.items() if k != 'temperature') or not 0 <= settings['temperature'] <= 2:
        raise ValueError("Budgets must be positive and temperature must be in [0,2].")
    if settings['max_output'] >= settings['context']:
        raise ValueError('max-output must be below context')
    task = required_task(args.tasks, instance_id)
    slot = args.out / model / args.config / str(args.seed) / instance_id
    slot.mkdir(parents=True, exist_ok=True)
    identity = dict(model=model, config=args.config, seed=args.seed, instance_id=instance_id,
                    settings=settings, base_commit=task['base_commit'], task_sha256=sha(stable(task)),
                    runner_sha256=sha(Path(__file__).read_bytes()), protocol_sha256=protocol_hash(settings),
                    model_revision=os.environ.get('HFIT_MODEL_REVISION', 'unspecified'),
                    image=image_name(task))
    with (slot/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = slot/'result.json'
        pred = slot/'prediction.json'
        trace = slot/'trace.jsonl'
        if result.exists():
            prev = json.loads(result.read_text())
            if any(prev.get(k) != v for k, v in identity.items()):
                raise ValueError(f'Existing slot was run under different settings or code: {slot}')
            if (prev['status'] != 'infra_error' and pred.exists() and trace.exists()
                and prev.get('prediction_sha256') == sha(pred.read_bytes()) and prev.get('trace_sha256') == sha(trace.read_bytes())):
                print('Verified existing', slot)
                return 0
        # Preserve incomplete/error evidence, including consumed tokens, before retrying.
        previous_files = [p for p in (result, trace, pred, slot/'verdict.json') if p.exists()]
        if previous_files:
            archive = slot/'previous'/str(time.time_ns())
            archive.mkdir(parents=True)
            for path in previous_files:
                os.replace(path, archive/path.name)
        stats = dict(identity, sampling=dict(top_p=0.95, top_k=20, enable_thinking=ENABLE_THINKING), status='infra_error', steps=0, prompt_tokens=0, completion_tokens=0,
                     diagnostics=dict(edits_seen=False, valid_test_runs=0, parse_errors=0, repeated_actions=0,
                                      restarts=0, submit_attempts=0))
        started = time.monotonic()
        with trace.open('w', encoding='utf-8', buffering=1) as f:
            def log(event, **fields):
                f.write(json.dumps(dict(event=event, elapsed_seconds=round(time.monotonic()-started,3), **fields),
                                   ensure_ascii=False) + '\n')
            try:
                llm = model_class(args.base_url, model, args.seed, settings['temperature'])
                state, patch = solve(task, args.config, settings, llm,
                                     lambda: env_class(task, settings['command_timeout']), log, stats)
                write_result(pred, dict(instance_id=instance_id, model_name_or_path=model, model_patch=patch))
                stats.update(status=state, patch_bytes=len(patch.encode()), prediction_sha256=sha(pred.read_bytes()))
                log('finish', status=state, patch_bytes=stats['patch_bytes'])
            except Exception as exc:
                stats['error'] = f'{type(exc).__name__}: {exc}'
                log('infra_error', detail=stats['error'])
        stats.update(trace_sha256=sha(trace.read_bytes()), elapsed_seconds=round(time.monotonic()-started, 3))
        write_result(result, stats)
        print(json.dumps({k:stats.get(k) for k in ('instance_id','status','patch_bytes','steps','error')}))
        return int(stats['status'] == 'infra_error')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks',type=Path,required=True)
    p.add_argument('--instance-id',required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--config',required=True)
    p.add_argument('--out',type=Path,default=Path('runs_thinking'))
    p.add_argument('--base-url',default=os.getenv('HFIT_BASE_URL','http://127.0.0.1:8000/v1'))
    for k, v in DEFAULTS.items():
        p.add_argument('--'+k.replace('_','-'),type=type(v),default=v)
    return p


if __name__ == '__main__':
    sys.exit(run_slot(parser().parse_args()))
