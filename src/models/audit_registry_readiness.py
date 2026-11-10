"""Read-only registry-readiness audit; no authentication, sync, model loading or upload."""
from __future__ import annotations

import ast
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import struct
import sys
from typing import Any, Iterator
import uuid
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from wandb.proto import wandb_internal_pb2 as pb
from src.utils.phase2_common import ROOT, load_params, output_path, sha256, write_json

LOGGER = logging.getLogger(__name__)


def stream_records(path: Path) -> Iterator[Any]:
    """Scan W&B log fragments read-only, validating headers, lengths and CRCs."""
    data = path.read_bytes()
    if data[:7] != b':W&B\xe1\xbe\x00':
        raise ValueError('Unsupported W&B stream header')
    offset, fragments = 7, []
    while offset < len(data):
        remaining = 32768 - offset % 32768
        if remaining < 7:
            if data[offset:offset + remaining] != bytes(remaining):
                raise ValueError('Invalid log padding')
            offset += remaining
            continue
        crc, length, kind = struct.unpack('<IHB', data[offset:offset + 7])
        offset += 7
        payload = data[offset:offset + length]
        offset += length
        if len(payload) != length or crc != zlib.crc32(payload, zlib.crc32(bytes([kind]))):
            raise ValueError('Corrupt or truncated W&B fragment')
        if kind == 1:
            if fragments:
                raise ValueError('Incomplete log record')
        elif kind == 2:
            if fragments:
                raise ValueError('Incomplete log record')
            fragments = [payload]
            continue
        elif kind == 3:
            if not fragments:
                raise ValueError('Missing first log fragment')
            fragments.append(payload)
            continue
        elif kind == 4:
            if not fragments:
                raise ValueError('Missing first log fragment')
            payload = b''.join(fragments + [payload])
            fragments = []
        else:
            raise ValueError('Unknown log fragment type')
        record = pb.Record()
        record.ParseFromString(payload)
        yield record
    if fragments:
        raise ValueError('Truncated final log record')


def main() -> None:
    """Audit exactly nine approved runs and the frozen bundle without deserialization."""
    params = load_params()
    cfg = params['phase2_registry']
    full = output_path(ROOT, cfg['full_cv_directory'])
    validation = output_path(ROOT, cfg['validation_directory'])
    final = output_path(ROOT, cfg['final_directory'])
    summary = json.loads((full / 'summary.json').read_text())
    decision = json.loads((validation / 'decision.json').read_text())
    completion = json.loads((final / 'completion.json').read_text())
    specifications = [(item['configuration'], Path(item['wandb_run_dir']).parent, summary['experiment_group'], params['phase2_experiments']['artifact_files']) for item in summary['configurations']]
    specifications += [(name + '-validation', Path(item['offline_wandb_run_dir']).parent, decision['experiment_group'], params['phase2_confirmation']['reviewed_artifact_files']) for name, item in decision['records'].items()]
    specifications += [('B0-reused-test', Path(completion['offline_wandb_run_directory']).parent, final.name, params['phase2_final']['reviewed_artifact_files'])]
    before = {}
    def checked(path: Path) -> str:
        """Remember every immutable input byte identity to detect audit side effects."""
        digest = sha256(path)
        before[str(path)] = digest
        return digest
    reports = []
    for expected_name, directory, group, allowlist in specifications:
        streams = list(directory.glob('run-*.wandb'))
        if len(streams) != 1:
            raise ValueError('Run must have exactly one transaction stream')
        stream = streams[0]
        checked(stream)
        counts, runs, artifacts, exits, files = Counter(), [], [], [], []
        for record in stream_records(stream):
            kind = record.WhichOneof('record_type')
            counts[kind] += 1
            if kind == 'run':
                run = record.run
                if run.display_name != expected_name or run.run_group != group or run.project != cfg['project']:
                    raise ValueError('Run identity/group mismatch')
                runs.append({'id': run.run_id, 'name': run.display_name, 'group': run.run_group, 'project': run.project, 'entity': run.entity})
            elif kind == 'exit':
                exits.append(record.exit.exit_code)
            elif kind == 'files':
                files.extend(item.path for item in record.files.files)
            elif kind == 'artifact':
                entries = []
                for entry in record.artifact.manifest.contents:
                    local = Path(entry.local_path)
                    if entry.path not in allowlist or entry.ref or not local.is_file() or not local.resolve().is_relative_to(ROOT.resolve()):
                        raise ValueError('Unexpected artifact or missing staged payload')
                    content = local.read_bytes()
                    if entry.size != len(content) or base64.b64encode(hashlib.md5(content).digest()).decode() != entry.digest:
                        raise ValueError('Staged artifact digest/size mismatch')
                    parsed = json.loads(content)
                    if not isinstance(parsed, (dict, list)):
                        raise ValueError('Unexpected staged artifact structure')
                    entries.append({'name': entry.path, 'size': len(content), 'sha256': checked(local), 'sdk_digest': entry.digest, 'staged_path': str(local.relative_to(ROOT))})
                artifacts.append({'name': record.artifact.name, 'type': record.artifact.type, 'digest': record.artifact.digest, 'entries': entries})
        if len(runs) != 1 or exits != [0] or len(artifacts) != 1:
            raise ValueError('Incomplete run or artifact record')
        if set(entry['name'] for entry in artifacts[0]['entries']) != set(allowlist):
            raise ValueError('Incomplete staged artifact allowlist')
        if set(files) != set(cfg['approved_extra_run_files']):
            raise ValueError('Unexpected auxiliary run files')
        for relative in files:
            checked(directory / 'files' / relative)
        if any(counts[kind] for kind in ['stats', 'environment', 'output', 'output_raw', 'output_logger']):
            raise ValueError('Unexpected environment, system or console capture')
        reports.append({'run': runs[0], 'directory': str(directory.relative_to(ROOT)), 'stream': str(stream.relative_to(ROOT)), 'stream_sha256': before[str(stream)], 'record_counts': dict(counts), 'exit_code': 0, 'artifacts': artifacts, 'auxiliary_files': files, 'synced_marker_present': bool(list(directory.glob('*.synced')))})
    bundle = final / params['phase2_final']['bundle_directory']
    bundle_manifest_path = bundle / params['phase2_final']['bundle_manifest_filename']
    expected_bundle_sha = completion['bundle_manifest_sha256']
    if checked(bundle_manifest_path) != expected_bundle_sha:
        raise ValueError('Frozen bundle manifest identity mismatch')
    manifest = json.loads(bundle_manifest_path.read_text())
    for relative, digest in manifest['files_sha256'].items():
        if checked(output_path(bundle, relative)) != digest:
            raise ValueError('Bundle file integrity mismatch')
    actual = {path.relative_to(bundle).as_posix() for path in bundle.rglob('*') if path.is_file()}
    expected = set(manifest['files_sha256']) | {params['phase2_final']['bundle_manifest_filename']}
    if actual != expected:
        raise ValueError('Unexpected or missing bundle files')
    original_source = (ROOT / params['phase2_final']['reference_class_source']).read_text(encoding='utf-8-sig')
    bundle_source = (bundle / params['phase2_final']['reference_class_source']).read_text(encoding='utf-8-sig')
    def class_body(source: str) -> str:
        """Compare exact class source without executing it."""
        node = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == params['phase2_final']['reference_class_name'])
        return ast.get_source_segment(source, node)
    if class_body(original_source) != class_body(bundle_source):
        raise ValueError('Compatibility class differs from original implementation')
    requirements = (bundle / params['phase2_final']['runtime_requirements_filename']).read_text().splitlines()
    if set(requirements) != {f'{name}=={version}' for name, version in manifest['runtime_dependencies'].items()}:
        raise ValueError('Runtime requirements differ from manifest')
    candidate = json.loads((bundle / manifest['candidate_manifest_file']).read_text())
    for relative, digest in candidate['artifacts_sha256'].items():
        if sha256(bundle / relative) != digest:
            raise ValueError('Bundle differs from selected B0 pair')
    changed = [path for path, digest in before.items() if sha256(Path(path)) != digest]
    if changed:
        raise RuntimeError('Audit modified immutable inputs')
    report = {'audit': 'Registry readiness only; no login, upload, sync, registration or model loading', 'runs': reports, 'run_count': len(reports), 'staged_entry_count': sum(len(artifact['entries']) for report in reports for artifact in report['artifacts']), 'bundle': {'directory': str(bundle.relative_to(ROOT)), 'manifest_sha256': expected_bundle_sha, 'files': sorted(actual), 'file_count': len(actual), 'all_sha256_valid': True, 'matches_selected_candidate': True, 'class_source_exact_match': True, 'runtime_requirements_match': True}, 'inputs_checked': len(before), 'changed_inputs': changed, 'open_requirements': ['Destination entity, organization, Registry permissions and private visibility are unconfirmed.', 'Existing runs have metadata lineage, not native use_artifact dependency edges.', 'Final config/evaluation metadata contain absolute local paths; review before upload.', 'Learned model vocabulary/feature names contain dataset-derived text; approve restricted sharing explicitly.', 'source_class_body_sha256 is mislabeled: it contains the original full source-file SHA256; do not rewrite frozen v1.0.0.', 'Fresh minimal runtime and FastAPI integration have not been tested.']}
    directory = output_path(ROOT, cfg['output_root']) / ('registry-readiness-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])
    directory.mkdir(exist_ok=False)
    write_json(directory / 'audit.json', report)
    LOGGER.info('Audited %d runs / %d staged entries / %d bundle files; %d immutable files unchanged; output=%s', len(reports), report['staged_entry_count'], len(actual), len(before), directory)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
