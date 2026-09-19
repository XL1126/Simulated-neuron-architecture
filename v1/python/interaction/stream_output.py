import sys


class StreamOutput:
    def __init__(self, callback=None):
        self.callback = callback
        self.buffer = ""

    def emit(self, char):
        self.write_char(char)

    def flush(self):
        sys.stdout.flush()

    def write_char(self, char):
        if char and len(char) > 0:
            self.buffer += char
            sys.stdout.write(char)
            sys.stdout.flush()
            if self.callback:
                try:
                    self.callback(char)
                except Exception:
                    pass

    def write(self, text):
        for ch in text:
            self.write_char(ch)

    def get_buffer(self):
        return self.buffer

    def clear(self):
        self.buffer = ""

    def newline(self):
        self.write_char('\n')
