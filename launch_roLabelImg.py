"""Launch in a child process so Python and native Qt failures can be logged."""
import datetime
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback


def open_log(root):
    name = 'roLabelImg-{}-{}.log'.format(
        datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f'), os.getpid())
    for directory in (root / 'logs', Path(tempfile.gettempdir()) / 'roLabelImg-logs'):
        try:
            directory.mkdir(parents=True, exist_ok=True)
            return directory / name, (directory / name).open('w', encoding='utf-8')
        except OSError as error:
            print('Cannot create log in {}: {}'.format(directory, error), flush=True)
    raise OSError('Could not create a diagnostic log in either location.')


def main():
    root = Path(os.path.abspath(__file__)).parent
    path, log = open_log(root)
    with log:
        def report(message):
            log.write(message + '\n')
            log.flush()
            # Keep logging even if the Windows console cannot encode a character.
            encoding = sys.stdout.encoding or 'utf-8'
            print(message.encode(encoding, errors='replace').decode(encoding), flush=True)

        report('Log file: {}'.format(path))
        report('Started: {}'.format(datetime.datetime.now().isoformat()))
        report('Python executable: {}'.format(sys.executable))
        report('Python version: {}'.format(sys.version.replace('\n', ' ')))
        report('Application directory: {}'.format(root))
        report('Arguments: {!r}'.format(sys.argv[1:]))
        for key in ('CONDA_PREFIX', 'VIRTUAL_ENV', 'QT_PLUGIN_PATH',
                    'QT_QPA_PLATFORM_PLUGIN_PATH', 'QT_QPA_PLATFORM', 'QT_DEBUG_PLUGINS'):
            report('{}: {}'.format(key, os.environ.get(key, '(unset)')))
        for module in ('PyQt5', 'PyQt4', 'lxml'):
            spec = importlib.util.find_spec(module)
            report('{} location: {}'.format(module, spec.origin if spec else '(not found)'))
        report('resources.py exists: {}'.format((root / 'resources.py').is_file()))
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        command = [sys.executable, '-u', '-X', 'faulthandler',
                   str(root / 'roLabelImg.py')] + sys.argv[1:]
        report('Launching: {!r}'.format(command))
        try:
            with subprocess.Popen(command, cwd=str(root), env=env,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as child:
                for line in iter(child.stdout.readline, b''):
                    report(line.decode('utf-8', errors='replace').rstrip('\r\n'))
                code = child.wait()
        except Exception:
            report(traceback.format_exc())
            return 1
        report('Process exit code: {} (0x{:08X})'.format(code, code & 0xffffffff))
        if code:
            report('Launch/application failed. Full output is saved in: {}'.format(path))
            report('For Qt plugin diagnostics, run "set QT_DEBUG_PLUGINS=1" in CMD, '
                   'then run Run_roLabelImg.bat again.')
        else:
            report('Application exited normally.')
        return code


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
