import sys
import threading
import queue
import os


class NonBlockingInput:
    def __init__(self):
        self.input_queue = queue.Queue()
        self._buffer = ""
        self._input_ended = False
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)

    def start(self):
        self._running = True
        self._thread.start()

    def stop(self):
        self._running = False

    def _read_loop(self):
        while self._running:
            try:
                line = sys.stdin.readline()
                if line:
                    self.input_queue.put(line.rstrip('\n'))
                else:
                    break
            except (OSError, ValueError, EOFError):
                break

    def read(self):
        if not self.input_queue.empty():
            self._input_ended = False
            text = self.input_queue.get_nowait()
            self._buffer = text
            return text
        return None

    def input_ended(self):
        return self._input_ended

    def mark_input_ended(self):
        self._input_ended = True

    def get_buffer(self):
        return self._buffer
