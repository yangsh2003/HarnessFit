"""Check the Docker save stream and the frozen-image import path with a fixture."""
import gzip
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import dataset_tools as dt


def sha(data):
    return hashlib.sha256(data).hexdigest()


def layer(value):
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        info = tarfile.TarInfo(value.decode())
        info.size = len(value)
        tar.addfile(info, io.BytesIO(value))
    return raw.getvalue()


class ImageImportTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.data = Path(temporary.name)
        self.archive = self.data / "environments/blobs/sha256"
        self.archive.mkdir(parents=True)
        self.args = SimpleNamespace(data=self.data, ids_file=None, wait_local=False,
                                    pull_missing=False, batch=True)
        self.pins = {}
        self.configs = {}
        self.chains = {}
        self.imports = []
        self.tags = {}
        layers = [layer(b"shared"), layer(b"first"), layer(b"second")]
        for i, issue in enumerate(("alpha", "beta"), 1):
            chain = (layers[0], layers[i])
            descriptors = [self.blob(gzip.compress(raw, mtime=0)) for raw in chain]
            config = self.blob(json.dumps({
                "architecture": "amd64", "os": "linux", "config": {},
                "rootfs": {"type": "layers", "diff_ids": ["sha256:" + sha(raw) for raw in chain]},
            }).encode())
            manifest = self.blob(json.dumps({
                "schemaVersion": 2, "config": config, "layers": descriptors
            }).encode())
            self.pins[issue] = "docker.io/example/" + issue + "@" + manifest["digest"]
            self.configs[issue] = config["digest"]
            self.chains[dt.tag(self.pins[issue])] = [p["digest"][7:] + ".tar.gz" for p in descriptors]
        self.start_patch("inventory", return_value=(self.pins, self.configs))
        self.start_patch("docker_id", side_effect=self.tags.get)
        self.start_patch("subprocess.run")

    def start_patch(self, name, **kwargs):
        mock = patch("dataset_tools." + name, **kwargs)
        self.addCleanup(mock.stop)
        return mock.start()

    def blob(self, content):
        name = sha(content)
        (self.archive / name).write_bytes(content)
        return {"digest": "sha256:" + name, "size": len(content)}

    def consume(self, archive, entries):
        output = io.BytesIO()
        dt.write_docker_archive(output, archive, entries)
        output.seek(0)
        with tarfile.open(fileobj=output) as tar:
            names = tar.getnames()
            self.assertEqual(len(names), len(set(names)))
            manifest = json.load(tar.extractfile("manifest.json"))
            self.imports.append(manifest)
            expected_names = {"manifest.json"}
            for record in manifest:
                ref, = record["RepoTags"]
                expected_names.update([record["Config"], *record["Layers"]])
                config_bytes = tar.extractfile(record["Config"]).read()
                self.tags[ref] = "sha256:" + sha(config_bytes)
                self.assertEqual(sha(config_bytes) + ".json", record["Config"])
                self.assertEqual(self.chains[ref], record["Layers"])
                diff_ids = []
                for name in record["Layers"]:
                    stored = tar.extractfile(name).read()
                    self.assertEqual(sha(stored) + ".tar.gz", name)
                    diff_ids.append("sha256:" + sha(gzip.decompress(stored)))
                self.assertEqual(diff_ids, json.loads(config_bytes)["rootfs"]["diff_ids"])
            self.assertEqual(expected_names, set(names))

    def test_batch_deduplicates_shared_layer_and_skips_existing_tags_on_retry(self):
        self.start_patch("docker_load", side_effect=self.consume)
        dt.images(self.args)
        dt.images(self.args)
        self.assertEqual([len(m) for m in self.imports], [2])
        self.assertEqual(3, len(set(self.chains["example/alpha:latest"] + self.chains["example/beta:latest"])))
        self.assertEqual(self.tags, {dt.tag(self.pins[k]): v for k, v in self.configs.items()})

    def test_default_still_emits_one_archive_per_missing_image(self):
        self.args.batch = False
        self.start_patch("docker_load", side_effect=self.consume)
        dt.images(self.args)
        self.assertEqual([len(m) for m in self.imports], [1, 1])

    def test_corrupt_second_image_aborts_before_batch_load(self):
        blob = self.archive / self.chains["example/beta:latest"][-1].split(".")[0]
        contents = blob.read_bytes()
        blob.write_bytes(bytes([contents[0] ^ 1]) + contents[1:])
        load = self.start_patch("docker_load")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            dt.images(self.args)
        load.assert_not_called()

    def test_missing_blob_aborts_before_batch_load(self):
        (self.archive / self.configs["beta"][7:]).unlink()
        load = self.start_patch("docker_load")
        with self.assertRaisesRegex(ValueError, "Missing local blobs for beta"):
            dt.images(self.args)
        load.assert_not_called()

    def test_wrong_config_after_load_fails_even_for_second_tag(self):
        def partial(archive, entries):
            self.consume(archive, entries[:-1])
        self.start_patch("docker_load", side_effect=partial)
        with self.assertRaisesRegex(ValueError, "example/beta:latest differs"):
            dt.images(self.args)
