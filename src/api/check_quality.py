"""Opt-in live API diagnostics: report quality failures without changing predictions."""
from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Any
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.phase2_common import load_params

LOGGER = logging.getLogger(__name__)


def request_prediction(url: str, payload: dict[str, str], timeout: float) -> tuple[int, Any]:
    """Send one real request, preserving validation errors and avoiding redirects."""
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        """Do not resend answer text to a different endpoint."""

        def redirect_request(self, *args: Any, **kwargs: Any) -> None:
            """Expose HTTP redirects as operational failures."""
            return None

    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    opener = urllib.request.build_opener(NoRedirect())
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def run(params: dict[str, Any]) -> int:
    """Return 0 for satisfied checks, 1 for quality failures, 2 for operational errors."""
    api = params['phase2_api']
    cfg = params['phase2_quality_diagnostics']
    url = f"http://{api['host']}:{api['port']}/predict"
    ordered_labels = [params['data']['target_classes'][key]
                      for key in sorted(params['data']['target_classes'])]
    results: dict[str, str] = {}
    operational_errors = False
    for name, answer in cfg['cases'].items():
        payload = dict(question=cfg['question'], reference_answer=cfg['reference_answer'],
                       candidate_answer=answer)
        try:
            status, body = request_prediction(url, payload, api['client_timeout_seconds'])
            if name == 'empty':
                if status != 422:
                    raise ValueError(f'Expected validation HTTP 422, received {status}')
                LOGGER.info('%s: HTTP %s; validation rejected empty input', name, status)
                continue
            if status != 200:
                raise ValueError(f'Expected HTTP 200, received {status}')
            if not isinstance(body, dict) or body.get('predicted_quality') not in ordered_labels:
                raise ValueError('Invalid predicted_quality response')
            if body.get('model_version') != api['expected_model_version']:
                raise ValueError('Unexpected model_version; review diagnostics for this version')
            results[name] = body['predicted_quality']
            LOGGER.info('%s: HTTP %s; predicted_quality=%s; model_version=%s',
                        name, status, body['predicted_quality'], body['model_version'])
        except (OSError, ValueError) as error:
            operational_errors = True
            LOGGER.error('%s: operational/contract failure (%s): %s', name, type(error).__name__, error)

    failed = False
    if 'correct' in results and 'incorrect' in results:
        if ordered_labels.index(results['correct']) <= ordered_labels.index(results['incorrect']):
            LOGGER.error('QUALITY FAIL: correct answer does not outrank reversed facts')
            failed = True
        else:
            LOGGER.info('QUALITY CHECK SATISFIED: correct answer outranks reversed facts')
    if 'nonsense' in results:
        if results['nonsense'] in cfg['nonsense_disallowed_labels']:
            LOGGER.error('QUALITY FAIL: unrelated nonsense received a high quality category')
            failed = True
        else:
            LOGGER.info('QUALITY CHECK SATISFIED: unrelated nonsense avoided high categories')
    if operational_errors:
        LOGGER.error('Diagnostics incomplete: operational/contract errors; exit 2')
        return 2
    if failed:
        LOGGER.error('Known model-quality failures reproduced; predictions unchanged; exit 1')
        return 1
    LOGGER.info('Diagnostic checks satisfied; this small example set does not establish model quality')
    return 0


def main() -> None:
    """Read all diagnostic inputs and connection settings from root params.yaml."""
    params = load_params()
    logging.basicConfig(level=params['base']['log_level'], format='%(levelname)s: %(message)s')
    raise SystemExit(run(params))


if __name__ == '__main__':
    main()
