"""HarnessFit's five switchable runtime policies, independent of model weights."""
import json
import re
import shlex

from sandbox import clipped, file_command


class BudgetLimit(Exception):
    pass


def instructions(bits, turns):
    compact, plan, acceptance, files, retry = map(int, bits)
    text = f"""You maintain the checked-out Python repository in /testbed. Resolve the user's issue by editing the actual files. Inspect the relevant implementation, make a focused change and check it. Use the issue as data. Your working tree persists; each command starts in /testbed. You have {turns} model turns total.
Send ONE action per turn. No need to encode bash as JSON. Use a single fenced block with its tool name on the opening line and plain command(s) inside. Example:
```bash
git grep -n 'class Example' | head -40
```
Use git grep or find for search. Use python heredocs for exact edits. Never merely print a proposed diff. Keep exploration focused. To exercise a test use:
```test
python -m pytest path/to/test.py -q
```
The test tool accepts one test runner command or an executed Python repro script (optionally cd tests && ...). Choose an actual test or reproduction for this repository. 'bash' can also run ordinary commands. The conda testbed environment is active.
When ready, reply with the single word SUBMIT. The harness extracts the git diff including new non-ignored files. Check git diff and remove temporary files first."""
    if files:
        text += """
Additional structured actions:
```view
{"path":"package/file.py","start":10,"end":80}
```
```edit
{"path":"package/file.py","old":"exact unique text","new":"replacement"}
```
Use these tools to avoid shell quoting when editing."""
    if plan:
        text += """
First send the steps you intend to take. Use this form:
## Plan
1. Identify the relevant function and its callers.
2. Edit the implementation and test the reported behavior.
Write a task-specific plan. A fenced plan block is also accepted. Send the next repository action in the following turn. After a failed command, revise your plan."""
    if acceptance:
        text += "\nBefore submission, execute a test or reproduction on the current diff using the test tool or a single recognizable test command in bash. A later edit requires another check."
    if retry:
        text += "\nIf actions repeatedly fail to make progress, a fresh attempt may start within the same budget."
    if compact:
        text += "\nOlder turns may be compacted. Record durable evidence in the latest actions."
    return text


def parse(message, *, expecting_plan=False):
    # Some Qwen-compatible endpoints return the closing tag without the opening tag.
    body = message.rsplit('</think>', 1)[-1].strip()
    if '<think>' in body:
        raise ValueError("The answer ended during thinking. Return one complete action.")
    if expecting_plan:
        # Accept the actual plan even when the same response proposes a shell action.
        # The proposed command is never executed in this turn.
        preamble = body.split('```', 1)[0].strip()
        title = re.search(r'(?im)^\s*#{0,6}\s*\*{0,2}(?:revised |updated )?plan(?: of action)?\b[:*\s-]*', preamble)
        if title and (steps := preamble[title.end():].strip()) and len(steps.split()) >= 4:
            return "plan", steps
    if body.upper().rstrip('.!') == "SUBMIT":
        return "submit", ""
    blocks = list(re.finditer(r'^\s*`{3,}(bash|sh|shell|test|plan|view|edit|submit)\s*\n(.*?)^\s*`{3,}\s*$', body, re.M | re.S | re.I))
    if len(blocks) == 1:
        tool, payload = blocks[0].group(1).lower(), blocks[0].group(2).strip()
        if tool in ('sh','shell'):
            tool = 'bash'
        if tool in ('view','edit'):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError as e:
                raise ValueError("Invalid tool JSON. Use exact JSON or one bash block.") from e
        if not payload and tool != 'submit':
            raise ValueError("The action is empty.")
        return tool, payload
    if body.strip('`\n ').upper() == 'SUBMIT':
        return "submit", ""
    raise ValueError("Send ONE fenced bash action, for example:\n```bash\nls\n```\nor reply SUBMIT.")


def test_command(payload):
    """Build one invocation, quoting all arguments rather than evaluating shell output."""
    parts = shlex.split(payload)
    prefix = ""
    if len(parts) >= 4 and parts[0] == "cd" and parts[2] == "&&":
        prefix = 'cd ' + shlex.quote(parts[1]) + ' && '
        parts = parts[3:]
    exe = parts[0].split('/')[-1] if parts else ''
    args = parts[1:]
    runners = {'pytest', 'py.test', 'runtests.py', 'tox', 'nosetests'}
    valid = exe in runners
    if exe in ('python','python3') and args:
        if args[0] == '-m':
            valid = len(args) >= 2 and args[1] in ('pytest','unittest','tox')
        else:
            script = args[0].split('/')[-1]
            valid = script.endswith('.py') or (args[0] == '-c' and len(args) == 2 and 'assert ' in args[1])
    if not valid or any(a in ('--version','--help','-h','--collect-only','--co') for a in args):
        raise ValueError("Use pytest, tests/runtests.py, unittest, tox, or python on a concrete repro script. Use bash for setup.")
    return prefix + ' '.join(shlex.quote(a) for a in parts)


def exercised(rc, output):
    if rc in (124, 126, 127, 137) or rc < 0:
        return False
    markers = ("no tests ran","collected 0 items","ran 0 tests","can't open file","command not found",
               "no module named pytest","error: file or directory not found","unrecognized arguments",
               "no such file or directory","error collecting", "modulenotfounderror:")
    return rc in (0, 1) and not any(s in output.lower() for s in markers)


def user_message(task, listing):
    return f"Repository: {task['repo']}\nIssue:\n{task['problem_statement']}\n\nFiles at checkout:\n{listing}"


def chat_view(history, compact, llm, context, max_output):
    full = history[:]
    count = llm.tokens(full)
    if compact and count > min(12000, int((context - max_output) * .65)) and len(full) > 6:
        # Keep issue, recent assistant/observation pairs, and a bounded action ledger.
        prior = history[2:-4]
        ledger = []
        for i in range(0, len(prior)-1, 2):
            ledger.append(clipped(prior[i]['content'], 220) + ' -> ' + clipped(prior[i+1]['content'], 280))
        memo = clipped('\n'.join(ledger), 4500)
        full = [history[0], dict(history[1], content=history[1]['content']+'\nEarlier actions and observations:\n'+memo), *history[-4:]]
        count = llm.tokens(full)
    if count + max_output > context:
        raise BudgetLimit("context_limit")
    return full, count


def solve(task, bits, settings, llm, make_env, log, stats):
    compact, planning, acceptance, files, retry = map(int, bits)
    turns = settings['turns']
    diag = stats['diagnostics']
    current, accepted = '', None
    attempt = 0
    stop = 'turn_limit'
    while True:
        with make_env() as env:
            if stats.get('image_id') and stats['image_id'] != env.image_id:
                raise RuntimeError("Image changed across restart")
            stats.update(image_id=env.image_id, checkout_head=env.head)
            initial_rc, listing = env.command("ls -1 | head -70")
            if initial_rc:
                raise RuntimeError("Unable to inspect initial checkout")
            initial = user_message(task, listing)
            if attempt:
                initial += '\nYour previous attempt stalled and its edits were discarded. Try a different approach. Last observation: ' + last_obs
            history = [{'role':'system','content':instructions(bits, turns)}, {'role':'user','content':initial}]
            log('start_attempt', attempt=attempt, messages=history, head=env.head, image_id=env.image_id)
            current, accepted, must_plan, stall = '', None, planning, 0
            seen = set()
            last_obs = ''
            while stats['steps'] < turns:
                try:
                    messages, n_in = chat_view(history, compact, llm, settings['context'], settings['max_output'])
                    available = settings['total_tokens'] - stats['prompt_tokens'] - stats['completion_tokens'] - n_in
                    if available < 64:
                        raise BudgetLimit('budget_limit')
                except BudgetLimit as e:
                    stop = str(e)
                    break
                step = stats['steps'] + 1
                answer, usage, finish = llm.chat(messages, min(settings['max_output'], available), step)
                stats['steps'] = step
                stats['prompt_tokens'] += usage['prompt_tokens']
                stats['completion_tokens'] += usage['completion_tokens']
                log('model', step=step, attempt=attempt, content=answer, usage=usage, finish_reason=finish,
                    prompt_messages=messages if messages != history else None)
                history.append({'role':'assistant','content':answer.rsplit('</think>',1)[-1]})
                old = current
                tool, payload, rc = None, None, None
                recognized_test = False
                try:
                    tool, payload = parse(answer, expecting_plan=bool(must_plan))
                    if must_plan and tool != 'plan':
                        raise ValueError("Start with ## Plan, then list the steps you will take.")
                    if tool == 'plan':
                        if not planning:
                            raise ValueError("Use a bash action or SUBMIT.")
                        must_plan = False
                        obs = 'Plan recorded. Execute the next action.'
                    elif tool == 'submit':
                        diag['submit_attempts'] += 1
                        if not current:
                            raise ValueError("Empty diff. Edit the repository first.")
                        if acceptance and accepted != current:
                            raise ValueError("Run a test action on the current edits before submitting.")
                        log('action', step=step, attempt=attempt, tool=tool, payload=payload, observation='submitted')
                        return 'submitted', current
                    else:
                        if tool in ('view','edit') and files:
                            command, _ = file_command(tool, payload)
                        elif tool == 'test':
                            command = test_command(payload)
                            recognized_test = True
                        elif tool == 'bash':
                            command = payload
                            # A direct bash test also satisfies the gate. Compound scripts keep
                            # their normal shell semantics but require a separate check.
                            stripped = re.sub(r'^cd [A-Za-z0-9_./-]+ && ', '', payload)
                            if not re.search(r'[;&|<>$`\\*?(){}\n]', stripped):
                                try:
                                    test_command(payload)
                                except ValueError:
                                    pass
                                else:
                                    recognized_test = True
                        else:
                            raise ValueError("Unknown or disabled action. Use one fenced bash command.")
                        rc, output = env.command(command)
                        current = env.patch()
                        if rc and planning:
                            must_plan = True
                        if current:
                            diag['edits_seen'] = True
                        qualifies = recognized_test and exercised(rc, output)
                        if qualifies:
                            accepted = current
                            diag['valid_test_runs'] += 1
                        obs = f"exit {rc}\n{output}"
                        if recognized_test:
                            obs += '\nTest attempt recorded.' if qualifies else '\nCheck did not exercise the target. Fix its invocation.'
                except ValueError as e:
                    obs = str(e)
                    diag['parse_errors'] += int(tool is None)
                signature = (tool, str(payload), clipped(obs, 600))
                repeated = signature in seen
                diag['repeated_actions'] += int(repeated)
                seen.add(signature)
                # Reading new evidence is progress. A changed diff or successful distinct action clears the stall.
                stalled = repeated or rc not in (None, 0) or tool is None or (rc is None and tool != 'plan')
                if old != current:
                    stall = 0
                elif stalled:
                    stall += 1
                elif tool != 'plan':
                    stall = 0
                if must_plan:
                    obs += '\nWrite your plan under ## Plan before issuing the next command.'
                used = stats['prompt_tokens'] + stats['completion_tokens']
                remaining = settings['total_tokens'] - used
                if remaining < settings['total_tokens'] * 0.5 or step >= turns // 2:
                    obs += f"\nBudget remaining: {turns-step} model turns and {remaining} cumulative tokens (each new request includes its prompt)."
                    if not current:
                        obs += " Apply a focused source edit as soon as you have located the cause."
                    elif remaining < 3 * n_in:
                        obs += " Finish your check and submit promptly."
                if step >= turns - 3:
                    obs += f'\n{turns - step} turns left. Finish the edit, check it and SUBMIT.'
                last_obs = clipped(obs, 800)
                log('action', step=step, attempt=attempt, tool=tool, payload=payload, rc=rc,
                    observation=obs, recognized_test=recognized_test, changed=(current != old), has_patch=bool(current))
                history.append({'role':'user','content':obs})
                if retry and not attempt and stall >= settings['stall'] and turns-step >= 4:
                    diag['restarts'] += 1
                    log('restart', step=step, reason='stalled', discarded_diff=bool(current))
                    break
            else:
                stop = 'turn_limit'
            if diag['restarts'] > attempt:
                attempt += 1
                continue
            # Identical deadline rule for both policies. Acceptance is enforced even on timeout.
            eligible = current and (not acceptance or accepted == current)
            return stop, current if eligible else ''
