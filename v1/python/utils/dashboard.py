import json
import time
import threading
from flask import Flask, jsonify, render_template_string

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="2">
    <title>SNA v4 Dashboard</title>
    <style>
        *{margin:0;padding:0;box-sizing:border-box}
        body{font-family:'Segoe UI', Consolas, monospace; background:#0a0a0f; color:#d0d0d0; padding:20px}
        h1{color:#00e5ff; font-size:24px; margin-bottom:10px}
        .sub{color:#666; margin-bottom:20px; font-size:12px}
        .grid{display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); gap:16px}
        .card{background:#111118; border:1px solid #1a1a2e; border-radius:8px; padding:16px}
        .card h2{font-size:14px; color:#00e5ff; text-transform:uppercase; margin-bottom:12px; letter-spacing:1px}
        .row{display:flex; justify-content:space-between; padding:4px 0; font-size:12px; border-bottom:1px solid #1a1a2e}
        .row .key{color:#888}
        .row .val{color:#0f0; font-weight:bold}
        .warn{color:#ff9800}
        .good{color:#4caf50}
        .bad{color:#f44336}
        .bar{background:#1a1a2e; height:8px; border-radius:4px; margin-top:4px}
        .bar .fill{height:100%; border-radius:4px; transition:width 0.5s}
        .fill-dop{background:linear-gradient(90deg,#0f0,#ff9800,#f44336)}
        .fill-energy{background:linear-gradient(90deg,#2196f3,#00e5ff)}
        .fill-coherence{background:linear-gradient(90deg,#9c27b0,#e91e63)}
        .log{max-height:300px;overflow-y:auto;font-size:11px;color:#666}
        .log .entry{padding:2px 0; border-bottom:1px solid #111; font-family:monospace}
    </style>
</head>
<body>
<h1>SNA v4 — {{.Self.Architecture}}</h1>
<div class="sub">
    Step: {{.Step}} | Neurons: {{.Neurons}} | Firing: {{.Firing}} | Converged: {{.Converged}} | Coherence: {{.Coherence}} | Free Energy: {{.FreeEnergy}}
</div>
<div class="grid">
    <div class="card">
        <h2>Internal State</h2>
        {{range $k, $v := .SelfState}}
        <div class="row"><span class="key">{{$k}}</span><span class="val">{{printf "%.3f" $v}}</span></div>
        <div class="bar"><div class="fill fill-dop" style="width:{{mul $v 100}}%"></div></div>
        {{end}}
    </div>
    <div class="card">
        <h2>Output</h2>
        <div class="row"><span class="key">chars produced</span><span class="val">{{.Output.CharsProduced}}</span></div>
        <div class="row"><span class="key">buffer size</span><span class="val">{{.Output.BufferSize}}</span></div>
        <div class="row"><span class="key">silence steps</span><span class="val">{{.Output.SilenceCount}}</span></div>
        <div class="row"><span class="key">text</span><span class="val" style="color:#00e5ff">{{.Output.Text}}</span></div>
    </div>
    <div class="card">
        <h2>WTA Convergence</h2>
        <div class="row"><span class="key">leader</span><span class="val">{{.WTA.Leader}}</span></div>
        <div class="row"><span class="key">converged</span><span class="val">{{.WTA.Converged}}</span></div>
        <div class="row"><span class="key">conclusion neurons</span><span class="val">{{.WTA.ConclusionNeurons}}</span></div>
        <div class="row"><span class="key">dopamine</span><span class="val">{{printf "%.3f" .Dopamine}}</span></div>
        <div class="bar"><div class="fill fill-dop" style="width:{{mul .Dopamine 100}}%"></div></div>
    </div>
    <div class="card">
        <h2>Memory</h2>
        <div class="row"><span class="key">instant buffer</span><span class="val">{{.Memory.InstantBuffer}}</span></div>
        <div class="row"><span class="key">STM slots</span><span class="val">{{.Memory.STMSlots}}</span></div>
        <div class="row"><span class="key">LTM items</span><span class="val">{{.Memory.LTMItems}}</span></div>
        <div class="row"><span class="key">consolidations</span><span class="val">{{.Memory.Consolidations}}</span></div>
        <div class="row"><span class="key">replays</span><span class="val">{{.Memory.Replays}}</span></div>
    </div>
    <div class="card">
        <h2>SNN Decoder</h2>
        <div class="row"><span class="key">decoder neurons</span><span class="val">1000</span></div>
        <div class="row"><span class="key">vocab size</span><span class="val">12000</span></div>
        <div class="row"><span class="key">stable?</span><span class="val">{{.Decoder.Stable}}</span></div>
        <div class="row"><span class="key">top candidates</span><span class="val">{{.Decoder.TopCandidates}}</span></div>
    </div>
    <div class="card">
        <h2>Coherence & Binding</h2>
        <div class="row"><span class="key">global coherence</span><span class="val">{{printf "%.4f" .Coherence}}</span></div>
        <div class="bar"><div class="fill fill-coherence" style="width:{{mul .Coherence 100}}%"></div></div>
        <div class="row"><span class="key">bound groups</span><span class="val">{{.BoundGroups}}</span></div>
        <div class="row"><span class="key">free energy</span><span class="val">{{printf "%.5f" .FreeEnergy}}</span></div>
        <div class="row"><span class="key">known facts</span><span class="val">{{.SelfState.known_facts}}</span></div>
    </div>
</div>
<div class="log" style="margin-top:16px">
    {{range .RecentEvents}}
    <div class="entry">[{{.Time}}] {{.Event}}</div>
    {{end}}
</div>
</body>
</html>'''


class Dashboard:
    def __init__(self, brain, host='0.0.0.0', port=5050):
        self.brain = brain
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.recent_events = []

        @self.app.route('/')
        def index():
            return self._render()

        @self.app.route('/api/status')
        def api_status():
            return jsonify(self.brain.get_status())

        self._setup_routes()

    def _setup_routes(self):
        pass

    def start(self):
        def _run():
            try:
                self.app.run(host=self.host, port=self.port, debug=False, use_reloader=False)
            except Exception as e:
                print(f"[Dashboard] Error: {e}")

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        print(f"[Dashboard] http://{self.host}:{self.port}")

    def _render(self):
        status = self.brain.get_status()
        self_desc = status.get('self', {})
        internal_state = self_desc.get('internal_state', {})
        output = status.get('output', {})
        memory = status.get('memory', {})
        decoder = status.get('decoder', {})

        top_candidates = []
        if decoder:
            sorted_chars = sorted(decoder.items(), key=lambda x: x[1], reverse=True)
            top_candidates = [f'{c}:{v:.3f}' for c, v in sorted_chars[:5]]

        ctx = {
            'Self': {'Architecture': self_desc.get('architecture', 'SNA_v4')},
            'Step': status['step'],
            'Neurons': status['neurons'],
            'Firing': status['firing'],
            'Converged': status.get('converged', False),
            'Dopamine': status.get('dopamine', 0.5),
            'Coherence': status.get('coherence', 0.0),
            'FreeEnergy': status.get('free_energy', 0.0),
            'BoundGroups': status.get('bound_groups', 0),
            'SelfState': {
                'arousal': internal_state.get('arousal', 0.5),
                'valence': internal_state.get('valence', 0.5),
                'certainty': internal_state.get('certainty', 0.5),
                'energy': internal_state.get('energy', 0.8),
                'curiosity': internal_state.get('curiosity', 0.5),
                'focus': internal_state.get('focus', 0.5),
                'known_facts': self_desc.get('known_facts', 0),
            },
            'Output': {
                'CharsProduced': output.get('chars_produced', 0),
                'BufferSize': output.get('buffer_size', 0),
                'SilenceCount': output.get('silence_count', 0),
                'Text': output.get('current_text', '')[-40:] + '...' if len(output.get('current_text', '')) > 40 else output.get('current_text', ''),
            },
            'WTA': {
                'Leader': status.get('wta_leader', -1),
                'Converged': status.get('converged', False),
                'ConclusionNeurons': len(status.get('output', {}).get('neurons', [])),
                'Dopamine': status.get('dopamine', 0.5),
            },
            'Memory': {
                'InstantBuffer': memory.get('instant_buffer', 0),
                'STMSlots': memory.get('stm_slots', 0),
                'LTMItems': memory.get('ltm_items', 0),
                'Consolidations': memory.get('consolidations', 0),
                'Replays': memory.get('replay_sequences', 0),
            },
            'Decoder': {
                'Stable': 'Yes' if output.get('chars_produced', 0) > 0 else 'Pending...',
                'TopCandidates': ', '.join(top_candidates) if top_candidates else '-',
            },
            'RecentEvents': self.recent_events[-20:],
        }

        inner = json.dumps(ctx)
        html = HTML_TEMPLATE.replace('{{.Self.Architecture}}', str(ctx['Self']['Architecture']))
        html = html.replace('{{.Step}}', str(ctx['Step']))
        html = html.replace('{{.Neurons}}', str(ctx['Neurons']))
        html = html.replace('{{.Firing}}', str(ctx['Firing']))
        html = html.replace('{{.Converged}}', str(ctx['Converged']))
        html = html.replace('{{.Dopamine}}', str(ctx['Dopamine']))
        html = html.replace('{{.Coherence}}', str(ctx['Coherence']))

        return html