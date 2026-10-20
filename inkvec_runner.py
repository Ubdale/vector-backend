"""Optional, disposable Inkvec CLI conversion with bounded native resources."""
import logging
import ctypes
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET

MAX_SVG_BYTES = 8 * 1024 * 1024
MAX_MEMORY_BYTES = 512 * 1024 * 1024


def _binary():
    configured = os.environ.get('INKVEC_BINARY')
    path = Path(configured) if configured else Path(__file__).resolve().parent / 'bin' / ('inkvec.exe' if os.name == 'nt' else 'inkvec')
    return path if path.is_file() else None


def _linux_limits(parent_pid):
    # Called before exec in the disposable conversion worker; no API threads live here.
    import resource
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_SVG_BYTES, MAX_SVG_BYTES))
    resource.setrlimit(resource.RLIMIT_AS, (MAX_MEMORY_BYTES, MAX_MEMORY_BYTES))
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'Cannot set parent-death signal')
    # Close the race where the conversion worker died before prctl was set.
    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signal.SIGKILL)


class _WindowsJob:
    """Launch suspended, attach to kill-on-close job, then resume execution."""
    def __init__(self):
        from ctypes import wintypes as w
        class Basic(ctypes.Structure):
            _fields_ = [('PerProcessUserTimeLimit', ctypes.c_int64), ('PerJobUserTimeLimit', ctypes.c_int64), ('LimitFlags', w.DWORD), ('MinimumWorkingSetSize', ctypes.c_size_t), ('MaximumWorkingSetSize', ctypes.c_size_t), ('ActiveProcessLimit', w.DWORD), ('Affinity', ctypes.c_size_t), ('PriorityClass', w.DWORD), ('SchedulingClass', w.DWORD)]
        class IO(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in ('ReadOperationCount','WriteOperationCount','OtherOperationCount','ReadTransferCount','WriteTransferCount','OtherTransferCount')]
        class Extended(ctypes.Structure):
            _fields_ = [('BasicLimitInformation', Basic), ('IoInfo', IO), ('ProcessMemoryLimit', ctypes.c_size_t), ('JobMemoryLimit', ctypes.c_size_t), ('PeakProcessMemoryUsed', ctypes.c_size_t), ('PeakJobMemoryUsed', ctypes.c_size_t)]
        self.api = ctypes.WinDLL('kernel32', use_last_error=True)
        self.api.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
        self.api.CreateJobObjectW.restype = w.HANDLE
        self.api.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
        self.api.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        self.api.CloseHandle.argtypes = [w.HANDLE]
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = Extended()
        limits.BasicLimitInformation.LimitFlags = 0x2000 | 0x100 | 0x200  # KILL_ON_JOB_CLOSE | PROCESS_MEMORY | JOB_MEMORY
        limits.ProcessMemoryLimit = MAX_MEMORY_BYTES
        limits.JobMemoryLimit = MAX_MEMORY_BYTES
        if not self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            self.close()
            raise ctypes.WinError(ctypes.get_last_error())

    def attach(self, process):
        if not self.api.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())
        # Popen closes the primary thread handle. Resume the process only after
        # it is inside the job; it cannot escape if this Python worker dies.
        native = ctypes.WinDLL('ntdll')
        native.NtResumeProcess.argtypes = [ctypes.c_void_p]
        native.NtResumeProcess.restype = ctypes.c_long
        if native.NtResumeProcess(int(process._handle)) != 0:
            raise OSError('Cannot resume isolated Inkvec process')

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None


def _stop(process):
    if process is not None:
        if process.poll() is None:
            if os.name != 'nt':
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
        process.wait(timeout=2)


def trace_inkvec(raw, width, height, timeout=45):
    """Return CLI-produced SVG or None; missing/failed engines are optional."""
    process = None
    job = None
    try:
        binary = _binary()
        if binary is None or not (1 <= width <= 2000 and 1 <= height <= 2000) or len(raw) != width * height * 4:
            return None
        timeout = min(45, max(.01, float(timeout)))
        if not (timeout <= 45):
            return None
        from PIL import Image
        with tempfile.TemporaryDirectory(prefix='inkvec-',dir=os.environ.get('INKVEC_JOB_TEMP')) as directory:
            source = Path(directory) / 'input.png'
            output = Path(directory) / 'output.svg'
            Image.frombytes('RGBA', (width, height), raw).save(source, format='PNG')
            arguments = [str(binary.resolve()), str(source), '-o', str(output), '--mode', 'quality', '--restore', 'off', '--sr', 'off', '--lossy', 'on', '--max-dim', '768', '--time-budget', '3']
            environment = {**os.environ, 'RAYON_NUM_THREADS': '1'}
            options = dict(stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=environment, shell=False)
            if os.name == 'nt':
                job = _WindowsJob()
                options['creationflags'] = subprocess.CREATE_NO_WINDOW | 0x4
            elif sys_platform_linux():
                parent_pid = os.getpid()
                options.update(start_new_session=True, preexec_fn=lambda: _linux_limits(parent_pid))
            else:
                # Parent-death containment is required, not an optional promise.
                return None
            try:
                process = subprocess.Popen(arguments, **options)
                if job is not None:
                    job.attach(process)
                deadline = time.monotonic() + timeout
                while process.poll() is None:
                    if time.monotonic() >= deadline or (output.exists() and output.stat().st_size > MAX_SVG_BYTES):
                        logging.warning('Optional Inkvec refinement exceeded its time or output limit')
                        return None
                    time.sleep(.05)
                if process.returncode != 0 or not output.is_file() or output.stat().st_size > MAX_SVG_BYTES:
                    logging.warning('Optional Inkvec refinement failed (exit %s); keeping initial trace',process.returncode)
                    return None
                data = output.read_bytes()
                if len(data) > MAX_SVG_BYTES:
                    return None
                svg = data.decode('utf-8')
                if ET.fromstring(svg).tag != '{http://www.w3.org/2000/svg}svg':
                    return None
                return svg
            finally:
                if job is not None:
                    job.close()
                _stop(process)
    except Exception as error:
        logging.warning('Optional Inkvec refinement unavailable (%s)',type(error).__name__)
        return None


def sys_platform_linux():
    import sys
    return sys.platform.startswith('linux')
