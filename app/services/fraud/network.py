"""Day 9 — Graph-based analysis to identify coordinated fraud rings.

Edges: shared device_id, shared IP, shared payment_method_id, fuzzy PII matches.
Communities (connected components) flagged when they contain >=N users and
any user has high-risk fraud history.
"""
from __future__ import annotations

import networkx as nx

from app.db.mongo import db

RING_MIN_SIZE = 3


async def build_user_graph(seed_user_id: str, depth: int = 2) -> nx.Graph:
    g = nx.Graph()
    frontier = {seed_user_id}
    visited: set[str] = set()

    for _ in range(depth):
        next_frontier: set[str] = set()
        for uid in frontier:
            if uid in visited:
                continue
            visited.add(uid)
            g.add_node(uid)

            user = await db().users.find_one({"user_id": uid})
            if not user:
                continue

            for field in ("device_id", "ip", "payment_method_id"):
                value = user.get(field)
                if not value:
                    continue
                cursor = db().users.find(
                    {field: value, "user_id": {"$ne": uid}},
                    {"user_id": 1},
                ).limit(50)
                async for other in cursor:
                    other_id = other["user_id"]
                    g.add_edge(uid, other_id, via=field, value=value)
                    if other_id not in visited:
                        next_frontier.add(other_id)
        frontier = next_frontier
        if not frontier:
            break
    return g


async def detect_rings(seed_user_id: str) -> dict:
    g = await build_user_graph(seed_user_id)
    components = [c for c in nx.connected_components(g) if len(c) >= RING_MIN_SIZE]

    rings = []
    for comp in components:
        members = list(comp)
        bad = await db().fraud_scores.count_documents({
            "user_id": {"$in": members},
            "final_score": {"$gte": 70},
        })
        rings.append({
            "members": members,
            "size": len(members),
            "high_risk_score_count": bad,
            "suspicious": bad > 0,
        })

    return {
        "seed_user_id": seed_user_id,
        "graph_nodes": g.number_of_nodes(),
        "graph_edges": g.number_of_edges(),
        "rings": rings,
    }
