"""CPU-only regression checks for harness decisions and exported Git patches.

The model answers and disposable repository are fixtures. No network, GPU, Docker,
benchmark instances or model weights are needed. Run: python3 -m unittest test_agent
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import agent
from sandbox import Sandbox, file_command


class GitFixture:
    """Runs the real export command in a per-attempt disposable Git repo."""
    patch = Sandbox.patch

    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        self.tools = tempfile.TemporaryDirectory()
        (Path(self.tools.name)/'python').symlink_to(sys.executable)
        self.image_id = 'fixture-v1'
        self.git('init', '-q')
        self.git('config', 'user.email', 'fixture@example.org')
        self.git('config', 'user.name', 'Fixture')
        (self.path / 'value.txt').write_text('wrong\n')
        (self.path / '.gitignore').write_text('scratch\n')
        (self.path / 'check.py').write_text("from pathlib import Path\nassert Path('value.txt').read_text() == 'fixed\\n'\n")
        self.git('add', '.')
        self.git('commit', '-qm', 'fixture base')
        self.head = self.git('rev-parse', 'HEAD').strip()

    def git(self, *args):
        return subprocess.run(['git', *args], cwd=self.path, check=True, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout

    def command(self, script, limit=10000, timeout=None):
        p = subprocess.run(['bash', '-c', script], cwd=self.path, timeout=timeout or 15,
                           env={**os.environ, 'PATH': self.tools.name + os.pathsep + os.environ['PATH']},
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return p.returncode, p.stdout

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.tools.cleanup()
        self.tmp.cleanup()


class Replies:
    def __init__(self, sequence):
        self.sequence = iter(sequence)
        self.messages = []

    def tokens(self, messages):
        return 30

    def chat(self, messages, max_output, step):
        self.messages.append([dict(m) for m in messages])
        return next(self.sequence), {'prompt_tokens': 30, 'completion_tokens': 15}, 'stop'


def run_fixture(sequence, config, stall=3):
    llm = Replies(sequence)
    stats = {'diagnostics': dict.fromkeys(('edits_seen', 'parse_errors', 'submit_attempts',
                                          'valid_test_runs', 'repeated_actions', 'restarts'), 0),
             'steps': 0, 'prompt_tokens': 0, 'completion_tokens': 0}
    settings = dict(turns=len(sequence), total_tokens=16384, max_output=256, context=4096, stall=stall)
    events = []
    def log(kind, **details):
        events.append((kind, details))
    result = agent.solve({'repo':'fixture/repo', 'problem_statement':'The value should be fixed.'},
                         config, settings, llm, GitFixture, log, stats)
    return result, events, stats


def action(name, body):
    return '```' + name + '\n' + body + '\n```'


FIX = action('bash', "printf 'fixed\\n' > value.txt")
TEST = action('test', 'python3 check.py')
PLAN = action('plan', 'Inspect the value, edit it, then check it.')


class AgentTests(unittest.TestCase):
    def test_format_recovery_to_patch(self):
        responses = ['I would fix the value.', '<think>an interrupted reply',
                     action('bash', 'pwd')+'\n'+action('bash','ls'),
                     '<think>hidden</think>\nHere is the edit.\n'+FIX, 'SUBMIT']
        (status, patch), _, stats = run_fixture(responses, '00000')
        self.assertEqual(status, 'submitted')
        self.assertIn('+fixed', patch)
        self.assertEqual(stats['diagnostics']['parse_errors'], 3)
        self.assertEqual(stats['steps'], len(responses))

    def test_planning_and_post_edit_validation(self):
        edit = action('edit', json.dumps({'path': 'value.txt', 'old': 'fixed', 'new': 'fixed-again'}))
        responses = [FIX, PLAN, action('bash', 'false'), FIX, PLAN, FIX,
                     'SUBMIT', TEST, edit, 'SUBMIT', TEST, PLAN, 'SUBMIT']
        (status, patch), events, stats = run_fixture(responses, '01110')
        self.assertEqual(status, 'submitted')
        self.assertIn('+fixed-again', patch)
        self.assertEqual(stats['diagnostics']['valid_test_runs'], 2)
        self.assertEqual(stats['diagnostics']['submit_attempts'], 3)
        responses = [e['observation'] for k,e in events if k=='action']
        self.assertGreaterEqual(sum('plan' in s.lower() for s in responses), 4)
        self.assertEqual(sum('Run a test action' in s for s in responses), 2)

    def test_markdown_plan_and_command_records_only_plan(self):
        bundled = '## Plan\n\n1. Read the current value.\n2. Correct it then test.' + '\n\n' + action('bash', 'exit 7')
        (status, patch), events, _ = run_fixture([bundled, FIX, 'SUBMIT'], '01000')
        first = next(e for k,e in events if k == 'action')
        self.assertEqual(first['tool'], 'plan')
        self.assertIsNone(first['rc'])
        self.assertEqual(status, 'submitted')
        self.assertIn('+fixed', patch)

    def test_bash_direct_test_satisfies_acceptance(self):
        (status, patch), _, stats = run_fixture([FIX, action('bash','python3 check.py'), 'SUBMIT'], '00100')
        self.assertEqual(status, 'submitted')
        self.assertIn('+fixed', patch)
        self.assertEqual(stats['diagnostics']['valid_test_runs'], 1)

    def test_deadline_and_unexecuted_tests(self):
        cases = [
            ([FIX], '00100', ''),
            ([FIX], '00000', '+fixed'),
            ([FIX, action('test', 'python3 missing.py')], '00100', ''),
            ([FIX, action('test', 'python3 check.py --help')], '00100', ''),
        ]
        for sequence, config, expected in cases:
            with self.subTest(sequence=sequence, config=config):
                (status, patch), _, _ = run_fixture(sequence, config)
                self.assertEqual(status, 'turn_limit')
                if expected:
                    self.assertIn(expected, patch)
                else:
                    self.assertEqual(patch, '')

    def test_retry_starts_clean_and_shares_turn_budget(self):
        seq = [action('bash', 'touch first_attempt'), 'bad format', 'bad format',
               action('bash','test ! -e first_attempt'), FIX, TEST, 'SUBMIT']
        (status, patch), events, stats = run_fixture(seq, '00001', stall=2)
        self.assertEqual(status, 'submitted')
        self.assertNotIn('first_attempt', patch)
        self.assertEqual(stats['diagnostics']['restarts'], 1)
        self.assertEqual(stats['steps'], len(seq))
        self.assertEqual(len([e for kind,e in events if kind=='start_attempt']), 2)

    def test_compaction_retains_issue_and_recent_observations(self):
        history = [{'role':'system','content':'system'}, {'role':'user','content':'unique issue'}]
        for step in range(10):
            history.extend([{'role':'assistant','content':'action'+str(step)},
                            {'role':'user','content':'observation'+str(step)+'x'*2000}])
        counter = mock.Mock()
        counter.tokens.side_effect = lambda messages: sum(len(x['content']) for x in messages)
        with self.assertRaises(agent.BudgetLimit):
            agent.chat_view(history, False, counter, 15000, 500)
        compacted, count = agent.chat_view(history, True, counter, 15000, 500)
        self.assertIn('unique issue', compacted[1]['content'])
        self.assertEqual(compacted[-4:], history[-4:])
        self.assertLess(count+500, 15000)

    def test_structured_edit_preserves_bytes_and_refuses_ambiguous_edits(self):
        with GitFixture() as env:
            (env.path / 'quoted name.txt').write_bytes(b'a\r\na\r\n')
            args = {'path': 'quoted name.txt', 'old': 'a', 'new': 'b'}
            rc, _ = env.command(file_command('edit', args)[0])
            self.assertNotEqual(rc, 0)
            args.update(old='a\r\na', new='b\r\nb')
            rc, _ = env.command(file_command('edit', args)[0])
            self.assertEqual(rc, 0)
            self.assertEqual((env.path / 'quoted name.txt').read_bytes(), b'b\r\nb\r\n')

    def test_patch_exports_staged_untracked_and_committed(self):
        with GitFixture() as env:
            (env.path / 'value.txt').write_text('committed\n')
            env.git('commit', '-qam', 'agent commit')
            (env.path / 'staged.txt').write_text('staged\n')
            env.git('add', 'staged.txt')
            (env.path / 'new file').write_text('untracked\n')
            (env.path / 'scratch').write_text('ignored\n')
            patch = env.patch()
            self.assertIn('+committed', patch)
            self.assertIn('+staged', patch)
            self.assertIn('+untracked', patch)
            self.assertNotIn('ignored', patch)
            self.assertEqual(env.git('diff', '--cached', '--name-only').strip(), 'staged.txt')
            env.git('reset', '--hard', env.head)
            env.git('clean', '-fdx')
            p = subprocess.run(['git','apply','-'], input=patch, cwd=env.path,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertEqual((env.path/'new file').read_text(), 'untracked\n')

    def test_image_initialization_with_single_setup_commit(self):
        # Exercise the exact startup Git script on a clean image-style worktree.
        # Container launch and local image identity are mocked, not a Docker test.
        with GitFixture() as env, tempfile.TemporaryDirectory() as other:
            base = env.head
            (env.path/'setup.file').write_text('environment marker\n')
            env.git('add', 'setup.file')
            env.git('commit', '-qm', 'image setup')
            image_head = env.git('rev-parse', 'HEAD').strip()
            task = {'base_commit':base, 'instance_id':'demo__demo-1'}
            def local(cmd, **kwargs):
                script = cmd.replace('/testbed', str(env.path)).replace('/tmp/hfit-base', other+'/repo')
                return env.command(script, **kwargs)
            with mock.patch('sandbox.docker') as docker, mock.patch('sandbox.image_name', return_value='fixture'):
                docker.return_value.stdout = 'sha256:fixture\n'
                connector = Sandbox(task, 30)
                with mock.patch.object(connector, 'command', side_effect=local):
                    with connector as box:
                        self.assertEqual(box.head, image_head)
                        self.assertEqual(env.git('rev-list','--count','HEAD').strip(), '1')
                        self.assertEqual(box.patch(), '')
                args = docker.call_args_list[0].args[0]
                self.assertIn('--network=none', args)
                self.assertEqual(docker.call_args_list[-1].args[0][:2], ['rm','-f'])


if __name__ == '__main__':
    unittest.main()
