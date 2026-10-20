"""Contain native failures and bound conversion time with a disposable process."""
import logging
import multiprocessing
import os
import tempfile


class TraceTimeout(RuntimeError):
    pass


class TraceFailed(RuntimeError):
    pass


def _trace_entry(connection, raw, width, height, settings, job_directory=None):
    try:
        # The API parent owns cleanup even if this worker is forcibly killed.
        if job_directory is not None:os.environ['INKVEC_JOB_TEMP']=job_directory
        # Native libraries live only in the disposable child, not the API worker.
        from tracer import trace
        connection.send(('ok', trace(raw, width, height, settings)))
    except Exception:
        logging.exception('Isolated tracing failed')
        connection.send(('error', 'Conversion failed'))
    finally:
        connection.close()


def _run_isolated(target, args, timeout):
    context = multiprocessing.get_context('spawn')
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=target, args=(sender, *args), daemon=True)
    try:
        process.start()
        sender.close()
        if not receiver.poll(timeout):
            raise TraceTimeout('Conversion exceeded its time limit')
        try:
            status, result = receiver.recv()
        except (EOFError, OSError) as error:
            raise TraceFailed('Conversion worker exited without a result') from error
        if status != 'ok':
            raise TraceFailed('Conversion worker failed')
        return result
    finally:
        sender.close()
        receiver.close()
        if process.pid is not None:
            process.join(.2)
            if process.is_alive():
                process.terminate()
                process.join(2)
            if process.is_alive():
                process.kill()
                process.join(2)
            process.close()


def run_trace_job(raw, width, height, settings, timeout=90):
    with tempfile.TemporaryDirectory(prefix='vector-job-') as directory:
        return _run_isolated(_trace_entry, (raw, width, height, settings, directory), timeout)
