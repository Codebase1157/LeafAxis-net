"""Union-Find with union by size and path compression.

Used by `splitting/groupaware_split.py` to collapse SHA-256, pHash and cosine
near-duplicate edges into connected components before partitioning.
"""
from __future__ import annotations


class UnionFind:
    """A disjoint-set over integer ids 0..n-1."""

    __slots__ = ("parent", "size", "n")

    def __init__(self, n: int):
        self.n = n
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # Path compression.
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> bool:
        """Return True if `a` and `b` were in different components (and are now merged)."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True

    def components(self) -> dict[int, list[int]]:
        """Return {root: [members]}."""
        out: dict[int, list[int]] = {}
        for i in range(self.n):
            out.setdefault(self.find(i), []).append(i)
        return out
