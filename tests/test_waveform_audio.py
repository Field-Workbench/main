"""Source-clock and audio lifecycle regression tests; no audio device required."""
import ast
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from fieldworkbench.waveform_audio import AUDIO_MONITOR_JS, source_audio_payload


class AudioSourceTests(unittest.TestCase):
    def test_time_weighted_dc_and_constant_silence(self):
        source = source_audio_payload([0, 1, 4], [1, 5, 99])
        self.assertEqual(source['mean'], 4)
        self.assertEqual(source['peak'], 3)
        dc = source_audio_payload([0, .1, .3], [5, 5, 100])
        self.assertEqual(dc['peak'], 0)

    def test_invalid_clock(self):
        for times, values in [([0, 0], [1, 2]), ([0, 1], [1]),
                              ([0, 1], [1, float('nan')])]:
            with self.assertRaises(ValueError):
                source_audio_payload(times, values)

    def test_exposure_retains_only_original_clock(self):
        source = {'id': 's', 'phase_offset_s': .25, 'playback_mode': 'once_per_step'}
        protocol = {'duration_s': 1800, 'sequence': {'loop': True, 'steps': [
            {'duration_s': 2, 'drive_groups': [{'source_id': 's'}]},
            {'duration_s': 1, 'drive_groups': []}]}}
        payload = source_audio_payload([0, .5, 1], [1, -1, 1], source=source, protocol=protocol)
        self.assertEqual(len(payload['values']), 3)
        self.assertEqual(payload['phase_s'], .25)
        self.assertEqual([s['active'] for s in payload['steps']], [True, False])
        self.assertNotIn('enabled', payload)

    @unittest.skipUnless(shutil.which('node'), 'Node.js required for browser logic tests')
    def test_javascript_source_and_lifecycle(self):
        script = AUDIO_MONITOR_JS + (Path(__file__).with_name('waveform_audio_checks.js')).read_text()
        subprocess.run(['node', '-e', script], check=True, capture_output=True, text=True)

    @unittest.skipUnless(shutil.which('node'), 'Node.js required for syntax check')
    def test_embedded_viewer_script_syntax(self):
        path = Path(__file__).parents[1] / 'fieldworkbench' / 'waveform_gl_view.py'
        tree = ast.parse(path.read_text())
        template = next(n.value.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'template' for t in n.targets))
        html = template.replace('__PAYLOAD__', '{}').replace('__AUDIO_MONITOR__', AUDIO_MONITOR_JS)
        script = html.split('<script>')[-1].split('</script>')[0]
        with tempfile.TemporaryDirectory() as tmp:
            filename = Path(tmp) / 'viewer.js'
            filename.write_text(script)
            subprocess.run(['node', '--check', str(filename)], check=True, capture_output=True)


if __name__ == '__main__':
    unittest.main()
