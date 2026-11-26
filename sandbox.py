"""One disposable SWE-bench instance per attempt. No host workspace is mounted."""
import base64
import json
import os
import re
import shlex
import subprocess
import uuid
from pathlib import Path, PurePosixPath


def clipped(data, limit=10000):
    if len(data) <= limit:
        return data
    return data[:limit // 3] + "\n... [output clipped] ...\n" + data[-(limit * 2 // 3):]


def docker(args, timeout=90, check=True):
    p = subprocess.run(["docker", *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       timeout=timeout)
    # Decode without universal-newline translation: CRLF can be significant in patches.
    p.stdout = p.stdout.decode("utf-8", errors="surrogateescape")
    p.stderr = p.stderr.decode("utf-8", errors="replace")
    if check and p.returncode:
        raise RuntimeError("Docker: " + clipped(p.stderr + p.stdout, 2000))
    return p


def image_name(task):
    slug = task['instance_id'].lower().replace('__', '_1776_')
    return os.environ.get('HFIT_IMAGE_TEMPLATE', 'swebench/sweb.eval.x86_64.{slug}:latest').format(
        slug=slug, instance_id=task['instance_id'])


class Sandbox:
    def __init__(self, task, timeout):
        self.task = task
        self.timeout = timeout
        self.name = "hfit-" + uuid.uuid4().hex
        self.image = image_name(task)
        self.image_id = ""
        self.head = ""

    def __enter__(self):
        try:
            docker(["run", "-d", "--pull=never", "--network=none", "--pids-limit=512", "--name", self.name,
                    "-w", "/testbed", self.image, "sleep", "24h"], timeout=1200)
            self.image_id = docker(["inspect", "-f", "{{.Image}}", self.name]).stdout.strip()
            locked = json.loads(Path(__file__).with_name("image_configs.json").read_text())
            expected = locked.get(self.task["instance_id"])
            if expected and self.image_id != expected:
                raise RuntimeError(f"Unexpected Docker image ID {self.image_id}; expected {expected}. Run dataset_tools.py images.")
            base = self.task['base_commit']
            if not re.fullmatch(r'[a-fA-F0-9]{40}', base):
                raise ValueError("Expected a full benchmark base commit")
            # Official images can have a setup commit directly above the requested revision.
            # Keep installed files and expose only the starting tree, including in git history.
            cmd = f"""set -eu
h=$(git rev-parse HEAD)
test "$h" = {base} || test "$(git rev-parse HEAD^)" = {base}
git diff --quiet HEAD
git init -q /tmp/hfit-base
git -C /tmp/hfit-base fetch -q --no-tags --depth 1 file:///testbed "$h"
git -C /tmp/hfit-base reset -q --mixed FETCH_HEAD
rm -rf /testbed/.git
mv /tmp/hfit-base/.git /testbed/.git
git config user.email agent@localhost
git config user.name HarnessFit
git diff --quiet HEAD
git rev-parse HEAD"""
            rc, out = self.command(cmd, limit=3000, timeout=180)
            if rc or not re.fullmatch(r'[0-9a-f]{40}', out.strip()):
                raise RuntimeError("Unexpected repository/image setup: " + clipped(out, 900))
            self.head = out.strip()
            # Some images contain setup-generated nonignored files. Export only changes
            # made by the agent, using the exact clean-image working tree as the baseline.
            rc, tree = self.command(index_script(self.head, "git write-tree"), timeout=90)
            if rc or not re.fullmatch(r'[0-9a-f]{40}', tree.strip()):
                raise RuntimeError("Cannot snapshot initial working tree")
            self.baseline_tree = tree.strip()
            return self
        except BaseException:
            self.close()
            raise

    def close(self):
        docker(["rm", "-f", self.name], timeout=90, check=False)

    def __exit__(self, *_):
        self.close()

    def command(self, command, limit=10000, timeout=None):
        secs = timeout or self.timeout
        # Keep command output bounded before it traverses the Docker connection.
        # Capture stderr in the container, so errors from docker exec remain identifiable.
        body = 'source /opt/miniconda3/bin/activate testbed || exit 125\n' + command
        wrapper = """f=$(mktemp) || exit 125
trap 'rm -f "$f"' EXIT
timeout -k 4 "$1" bash -c "$2" >"$f" 2>&1
rc=$?
n=$(wc -c <"$f")
if [ "$n" -le "$3" ]; then cat "$f"; else
  head -c "$(($3 / 3))" "$f"; printf '\\n... [output clipped] ...\\n'; tail -c "$(($3 * 2 / 3))" "$f"
fi
exit "$rc" """
        p = docker(["exec", "-w", "/testbed", self.name, "bash", "-c", wrapper, "hfit",
                    str(secs), body, str(limit)], timeout=secs + 30, check=False)
        if p.stderr or p.returncode == 125:
            raise RuntimeError("Sandbox environment error: " + clipped(p.stderr + p.stdout, 800))
        return p.returncode, p.stdout if limit >= 2_000_000 else p.stdout.encode("utf-8", "replace").decode()

    def patch(self):
        # git diff against the immutable start includes staged and committed edits.
        # An isolated copy of the index also includes newly created, non-ignored files.
        origin = getattr(self, 'baseline_tree', self.head)
        cmd = index_script(origin, f"git diff --no-ext-diff --cached --binary {origin}")
        rc, out = self.command(cmd, limit=2_000_000, timeout=90)
        if rc or len(out.encode()) > 1_990_000 or '\n... [output clipped] ...\n' in out:
            raise RuntimeError("Cannot export complete repository diff")
        return out


def index_script(origin, operation):
    return f"""set -eu
idx=$(mktemp)
rm "$idx"
trap 'rm -f "$idx"' EXIT
export GIT_INDEX_FILE="$idx"
git read-tree {origin}
git add -A -- .
{operation}"""


def file_command(action, args):
    """Structured tools run inside the same isolated environment as bash."""
    if not isinstance(args, dict):
        raise ValueError("Expected a JSON object.")
    path = args.get('path')
    if not isinstance(path, str) or not path or PurePosixPath(path).is_absolute() or '..' in PurePosixPath(path).parts:
        raise ValueError("Use a repository-relative path.")
    if action == 'view':
        start, end = args.get('start', 1), args.get('end', None)
        end = start + 99 if end is None and type(start) is int else end
        if type(start) is not int or type(end) is not int or not 1 <= start <= end <= start + 400:
            raise ValueError("Provide a positive start/end range of at most 401 lines.")
        return f"set -o pipefail; nl -ba -- {shlex.quote(path)} | sed -n '{start},{end}p'", False
    if not all(isinstance(args.get(k), str) for k in ('old', 'new')) or not args['old']:
        raise ValueError("Use nonempty exact 'old' text and replacement 'new'.")
    # Use stdlib compatible with the historical Python installed for the task.
    script = """import sys, json, base64, pathlib
a=json.loads(base64.b64decode(sys.argv[1]).decode('utf8'))
p=pathlib.Path(a['path'])
raw=p.read_bytes()
old=a['old'].encode('utf8')
if raw.count(old)!=1: sys.exit('edit refused: old must occur exactly once; view the file and retry')
p.write_bytes(raw.replace(old,a['new'].encode('utf8'),1))
print('edit applied')"""
    payload = base64.b64encode(json.dumps(args).encode()).decode()
    return 'python -c ' + shlex.quote(script) + ' ' + shlex.quote(payload), True
