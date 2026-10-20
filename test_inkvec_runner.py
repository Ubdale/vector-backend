import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'tmp/python-deps'))
import tempfile
import types
import unittest
from unittest.mock import patch,MagicMock
import inkvec_runner as runner

SVG='<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0L1 1Z"/></svg>'

class StubProcess:
    pid=54321
    def __init__(self,args,mode='success',**options):
        self.args=args;self.options=options;self.mode=mode;self.killed=False
        self.returncode=None if mode=='timeout' else 3 if mode=='crash' else 0
        self.directory=Path(args[1]).parent
        if mode not in ('timeout','crash','missing'):
            Path(args[3]).write_text('broken' if mode=='invalid' else SVG,encoding='utf8')
    def poll(self):return self.returncode
    def kill(self):self.killed=True;self.returncode=-9
    def wait(self,timeout=None):return self.returncode

class InkvecRunnerTests(unittest.TestCase):
    def setUp(self):
        self.base=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.base.cleanup)
        location=patch.object(tempfile,'tempdir',self.base.name)
        location.start();self.addCleanup(location.stop)

    def run_stub(self,mode='success'):
        processes=[];job=MagicMock()
        def launch(args,**kwargs):
            process=StubProcess(args,mode,**kwargs);processes.append(process);return process
        with patch.object(runner,'_binary',return_value=Path(__file__)),patch.object(runner.subprocess,'Popen',side_effect=launch),patch.object(runner,'_WindowsJob',return_value=job),patch.object(runner,'sys_platform_linux',return_value=False),patch.object(runner.time,'sleep'),patch.object(runner.time,'monotonic',side_effect=[0,100]):
            result=runner.trace_inkvec(bytes([0,0,0,255])*4,2,2,timeout=.01)
        return result,processes,job

    def test_success_uses_expected_cli_and_cleans_directory(self):
        if runner.os.name!='nt':self.skipTest('Windows stub launch contract')
        result,processes,job=self.run_stub()
        self.assertEqual(result,SVG);process=processes[0]
        self.assertFalse(process.directory.exists())
        self.assertFalse(process.options['shell'])
        self.assertEqual(process.options['env']['RAYON_NUM_THREADS'],'1')
        self.assertIn('--restore',process.args);self.assertIn('--lossy',process.args)
        self.assertIn('--time-budget',process.args)
        self.assertTrue(process.options['creationflags'] & 0x4)
        job.attach.assert_called_once_with(process);job.close.assert_called_once()

    def test_crash_invalid_and_missing_output_are_optional_failures(self):
        if runner.os.name!='nt':self.skipTest('Windows stub launch contract')
        for mode in ['crash','invalid','missing']:
            result,processes,job=self.run_stub(mode)
            self.assertIsNone(result);self.assertFalse(processes[0].directory.exists());job.close.assert_called_once()

    def test_timeout_kills_native_child_and_removes_temp_files(self):
        if runner.os.name!='nt':self.skipTest('Windows stub launch contract')
        result,processes,job=self.run_stub('timeout')
        self.assertIsNone(result);self.assertTrue(processes[0].killed)
        self.assertFalse(processes[0].directory.exists());job.close.assert_called_once()

    def test_missing_binary_and_invalid_input_never_launch(self):
        with patch.object(runner,'_binary',return_value=None),patch.object(runner.subprocess,'Popen') as launch:
            self.assertIsNone(runner.trace_inkvec(b'',2,2));launch.assert_not_called()
        with patch.object(runner,'_binary',return_value=Path(__file__)),patch.object(runner.subprocess,'Popen') as launch:
            self.assertIsNone(runner.trace_inkvec(b'',2,2));self.assertIsNone(runner.trace_inkvec(b'',2001,1));launch.assert_not_called()

    def test_output_limit_rejects_svg_before_reading(self):
        if runner.os.name!='nt':self.skipTest('Windows stub launch contract')
        with patch.object(runner,'MAX_SVG_BYTES',8):
            result,processes,job=self.run_stub()
        self.assertIsNone(result);self.assertFalse(processes[0].directory.exists())

    def test_job_attachment_failure_kills_suspended_process(self):
        if runner.os.name!='nt':self.skipTest('Windows stub launch contract')
        processes=[];job=MagicMock();job.attach.side_effect=OSError('blocked')
        def launch(args,**kwargs):
            process=StubProcess(args,'timeout',**kwargs);processes.append(process);return process
        with patch.object(runner,'_binary',return_value=Path(__file__)),patch.object(runner.subprocess,'Popen',side_effect=launch),patch.object(runner,'_WindowsJob',return_value=job):
            self.assertIsNone(runner.trace_inkvec(bytes([0,0,0,255])*4,2,2))
        self.assertTrue(processes[0].killed);self.assertFalse(processes[0].directory.exists());job.close.assert_called_once()

    def test_linux_launch_is_session_isolated_with_parent_death_hook(self):
        processes=[]
        def launch(args,**kwargs):
            process=StubProcess(args,'success',**kwargs);processes.append(process);return process
        platform=types.SimpleNamespace(name='posix',environ=runner.os.environ,getpid=lambda:1234,killpg=MagicMock())
        with patch.object(runner,'os',platform),patch.object(runner,'_binary',return_value=Path(__file__)),patch.object(runner,'sys_platform_linux',return_value=True),patch.object(runner.subprocess,'Popen',side_effect=launch),patch.object(runner,'_WindowsJob') as job:
            self.assertEqual(runner.trace_inkvec(bytes([0,0,0,255])*4,2,2),SVG)
        options=processes[0].options
        self.assertTrue(options['start_new_session']);self.assertTrue(callable(options['preexec_fn']))
        self.assertFalse(processes[0].directory.exists());job.assert_not_called()

    def test_launch_exception_still_closes_job_and_cleans_temp_directory(self):
        if runner.os.name!='nt':self.skipTest('Windows launch contract')
        job=MagicMock();directories=[]
        def launch(args,**kwargs):
            directories.append(Path(args[1]).parent)
            raise OSError('cannot execute')
        with patch.object(runner,'_binary',return_value=Path(__file__)),patch.object(runner,'_WindowsJob',return_value=job),patch.object(runner.subprocess,'Popen',side_effect=launch):
            self.assertIsNone(runner.trace_inkvec(bytes([0,0,0,255])*4,2,2))
        self.assertFalse(directories[0].exists());job.close.assert_called_once()

if __name__=='__main__':unittest.main()

