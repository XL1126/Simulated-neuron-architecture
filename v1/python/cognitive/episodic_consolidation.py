import numpy as np
from collections import deque
import time


class Episode:
    def __init__(self, episode_id):
        self.id = episode_id
        self.events = []
        self.salience = 0.5
        self.emotional_tag = 0.0
        self.consolidated = False
        self.replay_count = 0
        self.timestamp = time.time()
        self.key_concepts = set()
        self.outcome = None

    def add_event(self, event):
        self.events.append(event)
        if isinstance(event, dict):
            concepts = event.get('concepts', [])
            self.key_concepts.update(concepts)
            if event.get('reward') is not None:
                self.emotional_tag += event['reward'] * 0.3
                self.salience += abs(event['reward']) * 0.2

    def get_pattern(self):
        pattern = np.zeros(128)
        for i, ev in enumerate(self.events):
            if isinstance(ev, dict) and 'vector' in ev:
                pattern += np.array(ev['vector']).flatten()[:128] * (1.0 / (i + 1))
        norm = np.linalg.norm(pattern)
        if norm > 1e-10:
            pattern /= norm
        return pattern

    def size(self):
        return len(self.events)


class EpisodicConsolidation:
    def __init__(self, memory_layer, spa_bridge, capacity=100):
        self.memory = memory_layer
        self.bridge = spa_bridge
        self.capacity = capacity

        self.episodes = {}
        self.episode_queue = deque(maxlen=capacity)
        self.current_episode = None
        self.next_id = 0

        self.consolidation_schedule = deque()
        self.total_replays = 0
        self.consolidated_count = 0

    def start_episode(self):
        self.current_episode = Episode(self.next_id)
        self.next_id += 1
        return self.current_episode

    def end_episode(self, outcome=None):
        if self.current_episode:
            self.current_episode.outcome = outcome
            self.current_episode.salience = min(1.0,
                self.current_episode.salience + abs(self.current_episode.emotional_tag))
            self.episodes[self.current_episode.id] = self.current_episode
            self.episode_queue.append(self.current_episode.id)

            if self.current_episode.salience > 0.4:
                self.consolidation_schedule.append(self.current_episode.id)

            while len(self.episodes) > self.capacity:
                oldest = self.episode_queue[0]
                if oldest in self.episodes:
                    del self.episodes[oldest]
                self.episode_queue.popleft()

            ep = self.current_episode
            self.current_episode = None
            return ep

    def add_to_current(self, event):
        if self.current_episode:
            self.current_episode.add_event(event)

    def consolidate(self, max_episodes=3):
        consolidated = 0
        for _ in range(min(max_episodes, len(self.consolidation_schedule))):
            if not self.consolidation_schedule:
                break
            ep_id = self.consolidation_schedule.popleft()
            if ep_id in self.episodes:
                ep = self.episodes[ep_id]
                if not ep.consolidated and ep.salience > 0.3:
                    pattern = ep.get_pattern()
                    tag = f'episode_{ep_id}'
                    self.memory.store_pattern(pattern, tag=tag, strength=ep.salience)
                    self.memory.consolidate(tag, pattern, ep.salience)
                    ep.consolidated = True
                    self.consolidated_count += 1
                    consolidated += 1

        return consolidated

    def replay(self, population, num_replays=5):
        replayed = 0
        for ep_id in list(self.episode_queue)[-10:]:
            if ep_id in self.episodes:
                ep = self.episodes[ep_id]
                if ep.consolidated and ep.replay_count < 10:
                    events = ep.events[:20]
                    for ev in events:
                        if isinstance(ev, dict) and 'neuron_ids' in ev:
                            for nid in ev['neuron_ids'][:3]:
                                population.inject_spike(nid % population.size(), 0.05, 1)
                    ep.replay_count += 1
                    self.total_replays += 1
                    replayed += 1
        return replayed

    def extract_knowledge(self):
        knowledge = {}
        for ep_id, ep in self.episodes.items():
            if ep.consolidated:
                for concept in ep.key_concepts:
                    if concept not in knowledge:
                        knowledge[concept] = {'count': 0, 'total_salience': 0.0}
                    knowledge[concept]['count'] += 1
                    knowledge[concept]['total_salience'] += ep.salience
        return knowledge

    def get_summary(self):
        return {
            'total_episodes': len(self.episodes),
            'consolidated': self.consolidated_count,
            'total_replays': self.total_replays,
            'pending': len(self.consolidation_schedule),
            'current_ep_size': self.current_episode.size() if self.current_episode else 0,
        }