"""Exercise vLLM's HTTP contract, slot publication and resumption using a tiny Git fixture."""
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
import json
from pathlib import Path
import tempfile
from threading import Thread
import unittest

from run import parser, run_slot, sha
from test_agent import GitFixture, FIX


class RunnerTest(unittest.TestCase):
    def test_http_to_prediction_and_resume(self):
        prompts = []
        responses = iter([
            ({'content':None, 'reasoning':'PRIVATE_REASONING_FIXTURE'}, 'length'),
            ({'content':FIX, 'reasoning_content':'PRIVATE_REASONING_FIXTURE'}, 'stop'),
            ({'content':'<think>PRIVATE_REASONING_FIXTURE</think>SUBMIT'}, 'stop'),
        ])
        healthy = False

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                prompts.append(payload)
                if self.path == '/tokenize':
                    body = {'count':100}
                else:
                    if not healthy:
                        self.send_error(503)
                        return
                    assert self.path == '/v1/chat/completions'
                    message, finish = next(responses)
                    body = {'choices':[{'message':message,'finish_reason':finish}],
                            'usage':{'prompt_tokens':100,'completion_tokens':12,
                                     'completion_tokens_details':{'reasoning_tokens':7}}}
                data = json.dumps(body).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.addCleanup(server.server_close)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        fixture = lambda *args: GitFixture()
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            root = Path(tmp)
            tasks = root/'tasks.jsonl'
            tasks.write_text(json.dumps(dict(instance_id='toy-1', repo='toy/repo',
                base_commit='a'*40, problem_statement='Correct the value.', test_patch='PRIVATE_FIXTURE_GOLD')))
            args = parser().parse_args(['--out',tmp,'--tasks',str(tasks),'--instance-id','toy-1',
                '--model','tiny','--config','00000','--seed','8','--base-url',
                f'http://127.0.0.1:{server.server_port}/v1'])
            slot = root/'tiny/00000/8/toy-1'
            self.assertEqual(run_slot(args, env_class=fixture), 1)
            self.assertFalse((slot/'prediction.json').exists())
            self.assertEqual(json.loads((slot/'result.json').read_text())['status'], 'infra_error')
            healthy = True
            self.assertEqual(run_slot(args, env_class=fixture), 0)
            pred = json.loads((slot/'prediction.json').read_text())
            self.assertIn('+fixed', pred['model_patch'])
            result = json.loads((slot/'result.json').read_text())
            self.assertEqual(result['status'], 'submitted')
            self.assertEqual(result['prediction_sha256'], sha((slot/'prediction.json').read_bytes()))
            self.assertEqual(result['completion_tokens'], 36)
            self.assertEqual(result['prompt_tokens'], 300)
            self.assertIs(result['sampling']['enable_thinking'], True)
            self.assertEqual(result['diagnostics']['reasoning_turns'], 3)
            self.assertEqual(result['diagnostics']['reasoning_only_truncations'], 1)
            self.assertEqual(result['diagnostics']['length_limited_turns'], 1)
            trace = (slot/'trace.jsonl').read_text()
            self.assertNotIn('PRIVATE_REASONING_FIXTURE', trace)
            self.assertNotIn('PRIVATE_REASONING_FIXTURE', json.dumps(prompts))
            turns = [event for event in map(json.loads, trace.splitlines()) if event['event'] == 'model']
            self.assertTrue(all(turn['usage']['reasoning_tokens'] == 7 for turn in turns))
            self.assertEqual(turns[0]['content'], '')
            self.assertNotIn('PRIVATE_FIXTURE_GOLD', (slot/'trace.jsonl').read_text())
            self.assertNotIn('PRIVATE_FIXTURE_GOLD', json.dumps(prompts))
            self.assertTrue(all(p['chat_template_kwargs'] == {'enable_thinking':True} for p in prompts))
            num_calls = len(prompts)
            self.assertEqual(run_slot(args, env_class=fixture), 0)
            self.assertEqual(len(prompts), num_calls)
            args.turns = 3
            with self.assertRaisesRegex(ValueError, 'different settings'):
                run_slot(args, env_class=fixture)
