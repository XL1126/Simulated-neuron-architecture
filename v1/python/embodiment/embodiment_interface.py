import random


class EmbodimentInterface:
    def __init__(self, cfg=None):
        from .virtual_world import VirtualWorld
        from .terminal_emulator import TerminalEmulator
        from .web_search import WebSearchAPI

        if isinstance(cfg, dict):
            env_cfg = cfg.get('env_config', {})
            self.virtual_env = VirtualWorld()
            self.terminal = TerminalEmulator(enabled=cfg.get('terminal', True))
            self.web_search = WebSearchAPI(enabled=cfg.get('web_search', True))
        else:
            self.virtual_env = cfg if cfg else VirtualWorld()
            self.terminal = TerminalEmulator(enabled=True)
            self.web_search = WebSearchAPI(enabled=True)

        self.action_history = []
        self.learned_commands = {}
        self.pending_actions = []
        self.event_queue = []

    def process_thought(self, output_text):
        return self.step(output_text)

    def get_events(self):
        events = list(self.event_queue)
        self.event_queue.clear()
        return events

    def step(self, output_text):
        commands = self._extract_commands(output_text)

        feedback_spikes = []

        for cmd in commands:
            if not cmd:
                continue

            self.action_history.append({'command': cmd, 'time': len(self.action_history)})
            if len(self.action_history) > 1000:
                self.action_history.pop(0)

            result = self._execute_command(cmd)
            feedback = self._result_to_spikes(result, cmd)
            feedback_spikes.extend(feedback)
            self.event_queue.append({'type': 'embodiment', 'data': str(result)[:100]})

        return feedback_spikes

    def _extract_commands(self, text):
        if not text:
            return []

        import re
        commands = []

        action_pattern = re.findall(r'\[ACTION:(.*?)\]', text)
        if action_pattern:
            commands.extend(action_pattern)

        search_pattern = re.findall(r'\[SEARCH:(.*?)\]', text)
        for q in search_pattern:
            commands.append(f"search {q}")

        if not commands:
            parts = text.lower().split('\n')
            action_keywords = ['look', 'move', 'take', 'talk', 'search', 'ls', 'dir', 'pwd', 'echo']
            for part in parts:
                part = part.strip()
                for kw in action_keywords:
                    if part.startswith(kw):
                        commands.append(part)
                        break

        return commands

    def _execute_command(self, cmd):
        cmd = cmd.strip()

        if cmd.startswith("search "):
            query = cmd[7:].strip()
            return self.web_search.query(query)

        env_actions = ['look', 'move', 'take', 'talk', 'inventory', 'time']
        first_word = cmd.split()[0].lower() if cmd else ''

        if first_word in env_actions:
            return self.virtual_env.step(cmd)

        return self.terminal.execute(cmd)

    def _result_to_spikes(self, result, original_command):
        spikes = []

        if not result:
            return spikes

        for i, ch in enumerate(result):
            nid = (hash(ch) * (i + 1) * 31 + len(spikes)) % 10000
            spikes.append({
                'target_id': nid,
                'strength': 0.5,
                'time_ms': i,
                'char': ch,
                'group_hint': (hash(ch) + i) % 256
            })

        return spikes

    def get_learned_commands_summary(self):
        return self.action_history[-20:] if self.action_history else []
