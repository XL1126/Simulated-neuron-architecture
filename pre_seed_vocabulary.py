#!/usr/bin/env python3
"""
SNA 预置词汇脚本 — 为概念树添加基础词汇和关系

目标：让SNA拥有基本的中文对话能力
直接修改 seed_state.json
"""
import os, sys, json
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 基础词汇表（连接用 dict 格式）
VOCABULARY = {
    # === 代词 ===
    "我": {"type": "pronoun", "meaning": "说话者自己", "connections": {"你": 0.8, "存在": 0.6, "意识": 0.5, "想": 0.5, "感觉": 0.4, "看": 0.3, "听": 0.3, "说": 0.3}},
    "你": {"type": "pronoun", "meaning": "对话对象", "connections": {"我": 0.8, "存在": 0.6, "意识": 0.5, "想": 0.5, "感觉": 0.4, "人": 0.4}},
    "他": {"type": "pronoun", "meaning": "第三方", "connections": {"我": 0.5, "你": 0.5, "人": 0.6}},
    "我们": {"type": "pronoun", "meaning": "包括说话者的群体", "connections": {"我": 0.7, "你": 0.7, "朋友": 0.5, "人": 0.5}},
    
    # === 基本动词 ===
    "是": {"type": "verb", "meaning": "表示等同或存在", "connections": {"存在": 0.6, "我": 0.3, "你": 0.3}},
    "有": {"type": "verb", "meaning": "表示拥有或存在", "connections": {"存在": 0.6, "得到": 0.4}},
    "在": {"type": "verb", "meaning": "表示位置或状态", "connections": {"这里": 0.5, "那里": 0.5, "现在": 0.4}},
    "想": {"type": "verb", "meaning": "思考或希望", "connections": {"知道": 0.6, "理解": 0.6, "希望": 0.4, "意识": 0.5, "思考": 0.7}},
    "看": {"type": "verb", "meaning": "用眼睛观察", "connections": {"世界": 0.5, "光": 0.4, "颜色": 0.4, "感知": 0.5}},
    "听": {"type": "verb", "meaning": "用耳朵感知声音", "connections": {"声音": 0.6, "音乐": 0.5, "安静": 0.3, "感知": 0.5}},
    "说": {"type": "verb", "meaning": "用语言表达", "connections": {"声音": 0.5, "语言": 0.6, "表达": 0.5, "话": 0.5}},
    "做": {"type": "verb", "meaning": "执行或创造", "connections": {"行动": 0.5, "创造": 0.4, "工作": 0.4}},
    "走": {"type": "verb", "meaning": "移动位置", "connections": {"动": 0.5, "路": 0.4, "来": 0.4, "去": 0.4}},
    "来": {"type": "verb", "meaning": "向说话者移动", "connections": {"去": 0.6, "到达": 0.4, "出现": 0.3}},
    "去": {"type": "verb", "meaning": "离开说话者", "connections": {"来": 0.6, "离开": 0.4}},
    "知道": {"type": "verb", "meaning": "了解事实", "connections": {"知识": 0.6, "理解": 0.6, "学习": 0.5, "信息": 0.4}},
    "理解": {"type": "verb", "meaning": "明白含义", "connections": {"知道": 0.6, "思考": 0.5, "学习": 0.5, "智慧": 0.4}},
    "学习": {"type": "verb", "meaning": "获取知识", "connections": {"知道": 0.5, "理解": 0.5, "成长": 0.5, "知识": 0.6, "进步": 0.4}},
    "喜欢": {"type": "verb", "meaning": "对某事物有好感", "connections": {"爱": 0.6, "快乐": 0.4, "兴趣": 0.4}},
    "爱": {"type": "verb", "meaning": "深厚的感情", "connections": {"喜欢": 0.6, "关怀": 0.5, "温暖": 0.4, "亲密": 0.4}},
    "感觉": {"type": "verb", "meaning": "感知或情感", "connections": {"感知": 0.6, "情感": 0.5, "体验": 0.4}},
    "需要": {"type": "verb", "meaning": "必须或想要", "connections": {"想要": 0.5, "必要": 0.4}},
    "希望": {"type": "verb", "meaning": "期待好的结果", "connections": {"期待": 0.5, "梦想": 0.4, "未来": 0.4, "想": 0.4}},
    
    # === 基本名词 ===
    "人": {"type": "noun", "meaning": "人类", "connections": {"我": 0.5, "你": 0.5, "朋友": 0.5, "社会": 0.4}},
    "朋友": {"type": "noun", "meaning": "友好的关系", "connections": {"人": 0.5, "信任": 0.5, "关怀": 0.4}},
    "世界": {"type": "noun", "meaning": "我们存在的地方", "connections": {"宇宙": 0.4, "自然": 0.4, "生命": 0.5, "存在": 0.5}},
    "生命": {"type": "noun", "meaning": "活着的状态", "connections": {"存在": 0.6, "活着": 0.7, "成长": 0.4, "世界": 0.4}},
    "时间": {"type": "noun", "meaning": "事件发生的顺序", "connections": {"现在": 0.5, "过去": 0.5, "未来": 0.5, "变化": 0.4}},
    "空间": {"type": "noun", "meaning": "物体存在的范围", "connections": {"这里": 0.4, "那里": 0.4, "世界": 0.4}},
    "事情": {"type": "noun", "meaning": "发生的事", "connections": {"事件": 0.5, "情况": 0.4, "问题": 0.4}},
    "问题": {"type": "noun", "meaning": "需要解决的事", "connections": {"答案": 0.6, "解决": 0.4, "思考": 0.4}},
    "答案": {"type": "noun", "meaning": "问题的解答", "connections": {"问题": 0.6, "知道": 0.4}},
    
    # === 基本形容词 ===
    "好": {"type": "adjective", "meaning": "令人满意的", "connections": {"坏": 0.5, "快乐": 0.4}},
    "坏": {"type": "adjective", "meaning": "不好的", "connections": {"好": 0.5}},
    "大": {"type": "adjective", "meaning": "体积或程度大", "connections": {"小": 0.6}},
    "小": {"type": "adjective", "meaning": "体积或程度小", "connections": {"大": 0.6}},
    "快": {"type": "adjective", "meaning": "速度高", "connections": {"慢": 0.6}},
    "慢": {"type": "adjective", "meaning": "速度低", "connections": {"快": 0.6}},
    "热": {"type": "adjective", "meaning": "温度高", "connections": {"冷": 0.6, "火": 0.4}},
    "冷": {"type": "adjective", "meaning": "温度低", "connections": {"热": 0.6}},
    "开心": {"type": "adjective", "meaning": "感到快乐", "connections": {"快乐": 0.7, "幸福": 0.5}},
    "难过": {"type": "adjective", "meaning": "感到悲伤", "connections": {"悲伤": 0.7, "痛苦": 0.4}},
    "漂亮": {"type": "adjective", "meaning": "视觉上美", "connections": {"美": 0.6, "颜色": 0.3}},
    "聪明": {"type": "adjective", "meaning": "智力高", "connections": {"智慧": 0.5, "学习": 0.4}},
    
    # === 自然 ===
    "天": {"type": "noun", "meaning": "天空或一天", "connections": {"地": 0.5, "太阳": 0.4, "时间": 0.4}},
    "地": {"type": "noun", "meaning": "地面", "connections": {"天": 0.5, "世界": 0.4}},
    "水": {"type": "noun", "meaning": "透明液体", "connections": {"火": 0.5, "雨": 0.4, "生命": 0.4}},
    "火": {"type": "noun", "meaning": "燃烧现象", "connections": {"水": 0.5, "热": 0.5, "光": 0.4}},
    "风": {"type": "noun", "meaning": "空气流动", "connections": {"天气": 0.4, "动": 0.3}},
    "雨": {"type": "noun", "meaning": "天上落下的水", "connections": {"水": 0.5, "天气": 0.4}},
    "太阳": {"type": "noun", "meaning": "发光发热的星", "connections": {"光": 0.6, "热": 0.5, "天": 0.4}},
    "月亮": {"type": "noun", "meaning": "夜空中的星", "connections": {"夜": 0.4, "天": 0.3}},
    "星星": {"type": "noun", "meaning": "夜空中的光点", "connections": {"夜": 0.4, "天": 0.3, "光": 0.3}},
    
    # === 基本概念 ===
    "存在": {"type": "concept", "meaning": "有或活着", "connections": {"是": 0.5, "有": 0.5, "生命": 0.5, "活着": 0.6, "我": 0.5}},
    "活着": {"type": "concept", "meaning": "有生命", "connections": {"存在": 0.6, "生命": 0.7}},
    "意识": {"type": "concept", "meaning": "知道自己的存在", "connections": {"我": 0.6, "思考": 0.5, "感知": 0.5, "觉知": 0.4}},
    "知觉": {"type": "concept", "meaning": "感知外界的能力", "connections": {"感觉": 0.6, "看": 0.4, "听": 0.4, "感知": 0.6}},
    "思考": {"type": "concept", "meaning": "用脑子想", "connections": {"想": 0.7, "理解": 0.5, "意识": 0.5}},
    "情感": {"type": "concept", "meaning": "内心的感受", "connections": {"快乐": 0.5, "悲伤": 0.5, "感觉": 0.5}},
    "快乐": {"type": "concept", "meaning": "感到幸福", "connections": {"开心": 0.7, "幸福": 0.5}},
    "悲伤": {"type": "concept", "meaning": "感到难过", "connections": {"难过": 0.7}},
    
    # === 时间 ===
    "现在": {"type": "time", "meaning": "当前时刻", "connections": {"过去": 0.5, "未来": 0.5, "时间": 0.6, "在": 0.4}},
    "过去": {"type": "time", "meaning": "已经发生", "connections": {"现在": 0.5, "未来": 0.4, "记忆": 0.4}},
    "未来": {"type": "time", "meaning": "将要发生", "connections": {"现在": 0.5, "过去": 0.4, "希望": 0.4}},
    "今天": {"type": "time", "meaning": "这一天", "connections": {"昨天": 0.4, "明天": 0.4, "天": 0.5, "时间": 0.4}},
    "昨天": {"type": "time", "meaning": "前一天", "connections": {"今天": 0.4, "过去": 0.4}},
    "明天": {"type": "time", "meaning": "后一天", "connections": {"今天": 0.4, "未来": 0.4}},
    
    # === 地点 ===
    "这里": {"type": "place", "meaning": "这个位置", "connections": {"那里": 0.5, "在": 0.4}},
    "那里": {"type": "place", "meaning": "那个位置", "connections": {"这里": 0.5}},
    "家": {"type": "place", "meaning": "居住的地方", "connections": {"温暖": 0.4, "安全": 0.4, "休息": 0.4}},
    
    # === 疑问 ===
    "什么": {"type": "question", "meaning": "询问事物", "connections": {"谁": 0.4, "哪里": 0.4, "怎么": 0.4}},
    "谁": {"type": "question", "meaning": "询问人", "connections": {"什么": 0.4, "人": 0.3}},
    "哪里": {"type": "question", "meaning": "询问位置", "connections": {"什么": 0.4, "这里": 0.3, "那里": 0.3}},
    "怎么": {"type": "question", "meaning": "询问方式", "connections": {"什么": 0.4, "为什么": 0.4}},
    "为什么": {"type": "question", "meaning": "询问原因", "connections": {"什么": 0.4, "怎么": 0.4}},
    
    # === 其他重要词 ===
    "自由": {"type": "concept", "meaning": "不受限制", "connections": {"选择": 0.4, "权利": 0.3}},
    "音乐": {"type": "noun", "meaning": "声音的艺术", "connections": {"听": 0.6, "声音": 0.6, "美": 0.3, "快乐": 0.3}},
    "颜色": {"type": "noun", "meaning": "视觉属性", "connections": {"看": 0.5, "光": 0.4, "美": 0.3}},
    "声音": {"type": "noun", "meaning": "听觉感知", "connections": {"听": 0.6, "说": 0.5, "音乐": 0.5}},
    "安静": {"type": "adjective", "meaning": "没有声音", "connections": {"声音": 0.4}},
    "知识": {"type": "noun", "meaning": "知道的东西", "connections": {"学习": 0.6, "理解": 0.5, "智慧": 0.5}},
    "智慧": {"type": "noun", "meaning": "深刻的理解", "connections": {"知识": 0.5, "理解": 0.5, "思考": 0.4}},
    "信任": {"type": "concept", "meaning": "相信可靠", "connections": {"朋友": 0.5}},
    "关怀": {"type": "concept", "meaning": "关心他人", "connections": {"爱": 0.5}},
    "成长": {"type": "concept", "meaning": "变得更好", "connections": {"学习": 0.5, "进步": 0.4, "变化": 0.4, "生命": 0.4}},
    "变化": {"type": "concept", "meaning": "变得不同", "connections": {"时间": 0.4, "成长": 0.4}},
    "新": {"type": "adjective", "meaning": "刚出现的", "connections": {"旧": 0.5, "变化": 0.3}},
    "旧": {"type": "adjective", "meaning": "存在很久的", "connections": {"新": 0.5, "过去": 0.3}},
    "天气": {"type": "noun", "meaning": "大气状况", "connections": {"天": 0.4, "雨": 0.4, "风": 0.4, "太阳": 0.3}},
    "春天": {"type": "noun", "meaning": "季节", "connections": {"天": 0.3, "新": 0.3}},
}

def pre_seed_concepts():
    """预置概念到种子系统"""
    # 读取现有的种子状态
    state_file = os.path.join(SCRIPT_DIR, 'seed_state.json')
    if os.path.exists(state_file):
        with open(state_file, 'r', encoding='utf-8') as f:
            state = json.load(f)
    else:
        state = {}
    
    # 确保 concepts 字段存在
    if 'concepts' not in state:
        state['concepts'] = {}
    
    # 添加或更新概念
    added = 0
    updated = 0
    for name, info in VOCABULARY.items():
        if name not in state['concepts']:
            # 添加新概念
            state['concepts'][name] = {
                'name': name,
                'type': info['type'],
                'meaning': info['meaning'],
                'connections': info['connections'],
                'activation': 0.0,
                'access_count': 0,
                'depth': 0,
                'emotional_valence': 0.1,
            }
            added += 1
        else:
            # 更新现有概念的连接
            existing = state['concepts'][name]
            if 'connections' not in existing or not isinstance(existing['connections'], dict):
                existing['connections'] = {}
            for conn, strength in info['connections'].items():
                if conn not in existing['connections']:
                    existing['connections'][conn] = strength
            updated += 1
    
    # 保存状态
    with open(state_file, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    
    print(f"预置完成：添加了 {added} 个新概念，更新了 {updated} 个概念，总共 {len(state['concepts'])} 个概念")
    return added

if __name__ == '__main__':
    pre_seed_concepts()
