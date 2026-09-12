
STARTER_RULES = [
    {
        "name": "memory-pressure-response",
        "conditions": [{"signal": "memory_pct", "operator": ">", "value": 85}],
        "actions": [{"kind": "set", "param": "eviction_target_ratio", "value": 0.6, "delta": 0.0, "until": None}],
        "cooldown": 300,
        "priority": 5,
        "enabled": False,
        "last_fired_at": 0.0,
    },
    {
        "name": "attention-focusing",
        "conditions": [{"signal": "avg_similarity", "operator": "<", "value": 0.65}],
        "actions": [{"kind": "adjust", "param": "similarity_threshold", "value": None, "delta": 0.05, "until": 0.95}],
        "cooldown": 600,
        "priority": 5,
        "enabled": False,
        "last_fired_at": 0.0,
    },
    {
        "name": "knowledge-protection",
        "conditions": [{"signal": "eviction_count", "operator": ">", "value": 100}],
        "actions": [{"kind": "add", "param": "protected_kinds", "value": "fact", "delta": 0.0, "until": None}],
        "cooldown": 86400,
        "priority": 5,
        "enabled": False,
        "last_fired_at": 0.0,
    },
]
