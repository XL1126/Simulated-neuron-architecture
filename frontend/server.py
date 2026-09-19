#!/usr/bin/env python3
"""
SNA 聊天 API 服务器
独立后端，与 SNA 守护进程通过文件 IPC 通信。

端口: 8088
通信:
  - /tmp/sna_chat_in.jsonl  → 用户消息（本服务写入，SNA 读取）
  - /tmp/sna_chat_out.jsonl → 树的消息（SNA 写入，本服务读取）
  - /tmp/sna_chat_status.json → 树的状态（SNA 写入，本服务读取）
"""
import os
import json
import time
from flask import Flask, request, jsonify, send_file

app = Flask(__name__)

CHAT_IN = '/tmp/sna_chat_in.jsonl'
CHAT_OUT = '/tmp/sna_chat_out.jsonl'
STATUS_FILE = '/tmp/sna_chat_status.json'


def read_lines(filepath, since_byte=0):
    """读取文件中 since_byte 之后的新行"""
    if not os.path.exists(filepath):
        return [], since_byte
    try:
        sz = os.path.getsize(filepath)
        if sz <= since_byte:
            return [], since_byte
        with open(filepath, 'r', encoding='utf-8') as f:
            f.seek(since_byte)
            data = f.read()
            new_pos = sz
        lines = [l.strip() for l in data.strip().split('\n') if l.strip()]
        msgs = []
        for line in lines:
            try:
                msgs.append(json.loads(line))
            except:
                pass
        return msgs, new_pos
    except:
        return [], since_byte


@app.route('/')
def index():
    """返回聊天前端页面"""
    return send_file(os.path.join(os.path.dirname(__file__), 'chat.html'))


@app.route('/api/send', methods=['POST'])
def send_message():
    """用户发送消息到意识树"""
    data = request.json
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'empty message'}), 400

    msg = {
        'text': text,
        'time': time.time(),
        'time_str': time.strftime('%H:%M:%S'),
        'sender': 'user',
    }

    with open(CHAT_IN, 'a', encoding='utf-8') as f:
        f.write(json.dumps(msg, ensure_ascii=False) + '\n')

    return jsonify({'ok': True})


@app.route('/api/messages')
def get_messages():
    """获取新消息（轮询）"""
    since = int(request.args.get('since', 0))
    msgs, new_pos = read_lines(CHAT_OUT, since)
    return jsonify({'messages': msgs, 'pos': new_pos})


@app.route('/api/status')
def get_status():
    """获取意识树状态"""
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE) as f:
                return jsonify(json.load(f))
        except:
            pass
    return jsonify({'status': 'unknown'})


@app.route('/api/block', methods=['POST'])
def block():
    """拉黑/解除拉黑"""
    data = request.json
    blocked = data.get('blocked', True)
    cmd = '/block' if blocked else '/unblock'
    msg = {
        'text': cmd,
        'time': time.time(),
        'time_str': time.strftime('%H:%M:%S'),
        'sender': 'user',
    }
    with open(CHAT_IN, 'a', encoding='utf-8') as f:
        f.write(json.dumps(msg, ensure_ascii=False) + '\n')
    return jsonify({'ok': True, 'blocked': blocked})


if __name__ == '__main__':
    print("[SNA 聊天API] 启动于 http://0.0.0.0:8088")
    print(f"[SNA 聊天API] 消息输入: {CHAT_IN}")
    print(f"[SNA 聊天API] 消息输出: {CHAT_OUT}")
    app.run(host='0.0.0.0', port=8088, debug=False)
