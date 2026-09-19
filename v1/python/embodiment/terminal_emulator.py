import subprocess
import os
import tempfile


class TerminalEmulator:
    def __init__(self, enabled=True, safe_mode=True, max_output_length=4096):
        self.enabled = enabled
        self.safe_mode = safe_mode
        self.max_output_length = max_output_length
        self.working_dir = os.getcwd()
        self._blocked_commands = {'rm', 'sudo', 'chmod', 'chown', 'kill', 'shutdown',
                                   'reboot', 'format', 'mkfs', 'dd', 'fdisk'}

    def execute(self, command):
        if not self.enabled:
            return "[终端不可用]"

        if self.safe_mode:
            first_word = command.strip().split()[0].lower() if command.strip() else ''
            if first_word in self._blocked_commands:
                return "[命令已阻止: 安全模式]"

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.working_dir
            )

            output = result.stdout
            if result.stderr:
                output += "\n[STDERR]: " + result.stderr

            if len(output) > self.max_output_length:
                output = output[:self.max_output_length] + "\n...(输出已截断)"

            return output if output.strip() else "[命令执行完毕，无输出]"

        except subprocess.TimeoutExpired:
            return "[命令超时]"
        except Exception as e:
            return f"[终端错误]: {e}"

    def change_directory(self, path):
        try:
            new_path = os.path.join(self.working_dir, path)
            if os.path.isdir(new_path):
                self.working_dir = os.path.abspath(new_path)
                return f"当前目录: {self.working_dir}"
            return "目录不存在"
        except Exception as e:
            return f"[目录错误]: {e}"
