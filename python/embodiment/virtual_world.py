import random
import numpy as np
import math


class VirtualWorld:
    def __init__(self, width=8, height=8):
        self.width = width
        self.height = height
        self.agent_x = width // 2
        self.agent_y = height // 2
        self.agent_dir = 0

        self.objects = []
        self.colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'white', 'black']
        self.shapes = ['square', 'circle', 'triangle', 'diamond']
        self.tasks = [
            {'target_color': 'blue', 'target_shape': 'square', 'reward': '+1', 'done': False},
            {'target_color': 'red', 'target_shape': 'circle', 'reward': '+1', 'done': False},
            {'target_color': 'green', 'target_shape': 'diamond', 'reward': '+1', 'done': False},
        ]
        self.current_task = 0
        self.task_completed = 0
        self.task_attempts = 0
        self.max_attempts_per_task = 50

        self.total_reward = 0.0
        self.step_count = 0
        self.last_reward = 0.0

        self._spawn_objects()
        self._spawn_target()

    def _spawn_objects(self):
        self.objects.clear()
        for _ in range(max(3, (self.width * self.height) // 4)):
            obj = {
                'x': random.randint(0, self.width - 1),
                'y': random.randint(0, self.height - 1),
                'color': random.choice(self.colors),
                'shape': random.choice(self.shapes),
                'is_target': False,
            }
            self.objects.append(obj)

    def _spawn_target(self):
        if self.current_task >= len(self.tasks):
            return
        task = self.tasks[self.current_task]
        for _ in range(20):
            tx = random.randint(0, self.width - 1)
            ty = random.randint(0, self.height - 1)
            if abs(tx - self.agent_x) + abs(ty - self.agent_y) > 1:
                target = {
                    'x': tx, 'y': ty,
                    'color': task['target_color'],
                    'shape': task['target_shape'],
                    'is_target': True,
                }
                self.objects.append(target)
                return

    def get_features(self):
        features = np.zeros(16, dtype=np.float32)
        features[0] = self.agent_x / max(1, self.width - 1)
        features[1] = self.agent_y / max(1, self.height - 1)
        features[2] = self.agent_dir / 3.0

        objects_here = [o for o in self.objects
                        if o['x'] == self.agent_x and o['y'] == self.agent_y]
        for o in objects_here:
            ci = self.colors.index(o['color']) if o['color'] in self.colors else 0
            si = self.shapes.index(o['shape']) if o['shape'] in self.shapes else 0
            features[4 + ci] += 0.5 if not o['is_target'] else 1.0
            features[12 + si] = 1.0 if o['is_target'] else 0.0

        if self.current_task < len(self.tasks):
            task = self.tasks[self.current_task]
            tc = self.colors.index(task['target_color'])
            features[4 + tc] += 0.3

        return features

    def step(self, action_idx):
        self.step_count += 1
        self.last_reward = 0.0

        actions = ['wait', 'move_up', 'move_down', 'move_left', 'move_right',
                   'grab', 'look']
        action = actions[action_idx % len(actions)]

        if action == 'move_up':
            self.agent_y = max(0, self.agent_y - 1)
        elif action == 'move_down':
            self.agent_y = min(self.height - 1, self.agent_y + 1)
        elif action == 'move_left':
            self.agent_x = max(0, self.agent_x - 1)
        elif action == 'move_right':
            self.agent_x = min(self.width - 1, self.agent_x + 1)
        elif action == 'grab':
            objects_here = [o for o in self.objects
                            if o['x'] == self.agent_x and o['y'] == self.agent_y]
            for o in objects_here:
                if o['is_target'] and self.current_task < len(self.tasks):
                    self.objects.remove(o)
                    self.last_reward = 1.0
                    self.total_reward += 1.0
                    self.tasks[self.current_task]['done'] = True
                    self.task_completed += 1
                    self.current_task += 1
                    self._spawn_target()
                    self.task_attempts = 0

        if self.current_task < len(self.tasks):
            self.task_attempts += 1
            if self.task_attempts >= self.max_attempts_per_task:
                self.task_attempts = 0
                self.current_task += 1
                self._spawn_target()

        return {
            'action': action,
            'reward': self.last_reward,
            'position': (self.agent_x, self.agent_y),
            'features': self.get_features(),
            'task_completed': self.task_completed,
            'current_task': self.current_task,
        }

    def get_status(self):
        task_desc = ''
        if self.current_task < len(self.tasks):
            t = self.tasks[self.current_task]
            task_desc = f'找到{t["target_color"]} {t["target_shape"]}'
        return f'pos=({self.agent_x},{self.agent_y}) task={task_desc} reward={self.total_reward} done={self.task_completed}'

    def is_all_done(self):
        return self.current_task >= len(self.tasks)

    def to_spike_events(self):
        feats = self.get_features()
        events = []
        for i, v in enumerate(feats):
            if abs(v) > 0.01:
                nid = i * 7 + int(abs(v) * 50)
                events.append({'target_id': nid, 'strength': min(2.0, abs(v) * 3.0),
                                'time_ms': 0, 'source': 'virtual_world'})
        return events


class VirtualWorldEpisodeRunner:
    def __init__(self, virtual_world, population, spa_mapper=None):
        self.world = virtual_world
        self.pop = population
        self.spa_mapper = spa_mapper
        self.history = []
        self.dopamine_trace = []
        self.success_rate = 0.0
        self.total_actions = 0

    def run_episode(self, max_steps=200):
        self.world = VirtualWorld(self.world.width, self.world.height)
        episode_data = {'steps': [], 'total_reward': 0.0, 'success': False}

        for s in range(max_steps):
            features = self.world.get_features()

            spike_events = self.world.to_spike_events()
            for ev in spike_events:
                self.pop.inject_spike(ev['target_id'] % self.pop.size(),
                                      ev.get('strength', 1.0), 1)

            fires = self.pop.get_current_fires()
            active_neurons = [f.neuron_id for f in fires[:10]]
            action_idx = (sum(active_neurons) if active_neurons
                          else random.randint(0, 6)) % 7

            result = self.world.step(action_idx)
            self.total_actions += 1

            episode_data['steps'].append({
                'step': s, 'action': result['action'],
                'reward': result['reward'], 'pos': result['position'],
            })
            episode_data['total_reward'] += result['reward']

            if result['reward'] > 0:
                episode_data['success'] = True
                if self.spa_mapper:
                    task = self.world.tasks[self.world.current_task - 1] \
                        if self.world.current_task > 0 else None
                    if task:
                        color_neurons = self.spa_mapper.get_concept_neurons(task['target_color'])
                        shape_neurons = self.spa_mapper.get_concept_neurons(task['target_shape'])
                        action_neurons = self.spa_mapper.get_concept_neurons(result['action'])
                        self.spa_mapper.learn_from_pair(color_neurons, action_neurons, 0.3)
                        self.spa_mapper.learn_from_pair(shape_neurons, action_neurons, 0.3)
                        self.spa_mapper.reinforce_concept(task['target_color'], 0.1)

        self.history.append(episode_data)
        successes = sum(1 for h in self.history if h['success'])
        self.success_rate = successes / max(1, len(self.history))
        return episode_data