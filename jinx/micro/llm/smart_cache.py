"""Smart Cache - ML-enhanced caching for LLM responses with semantic similarity.

Uses embeddings for semantic caching and brain systems for intelligent eviction.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

_SMART_CACHE_MB = 100
_SMART_CACHE_ENTRIES = 1000

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    embedding: Optional[List[float]]
    hits: int
    created_at: float
    last_accessed: float
    size_bytes: int
    quality_score: float


class SmartCache:
    """ML-enhanced semantic cache for LLM responses."""
    
    def __init__(self, max_size_mb: int = 100, max_entries: int = 1000):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_entries = max_entries
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._embeddings_map: Dict[str, List[float]] = {}
        
        self._current_size = 0
        self._hits = 0
        self._misses = 0
        
        self._lock = asyncio.Lock()
    
    async def get(
        self,
        key: str,
        *,
        semantic: bool = True,
        similarity_threshold: float = 0.85
    ) -> Optional[Any]:
        """Get from cache with optional semantic search."""
        async with self._lock:
            # Exact match
            if key in self._cache:
                entry = self._cache[key]
                entry.hits += 1
                entry.last_accessed = time.time()
                self._hits += 1
                
                # Move to end (LRU)
                self._cache.move_to_end(key)
                
                return entry.value
            
            # Semantic search if enabled
            if semantic:
                similar = await self._find_similar(key, similarity_threshold)
                if similar:
                    self._hits += 1
                    return similar.value
            
            self._misses += 1
            return None
    
    async def put(
        self,
        key: str,
        value: Any,
        *,
        quality_score: float = 0.8,
        compute_embedding: bool = True
    ) -> bool:
        """Put value in cache with ML enhancement."""
        async with self._lock:
            # Compute entry size
            size_bytes = len(str(value))
            
            # Compute embedding if enabled
            embedding = None
            if compute_embedding:
                embedding = await self._compute_embedding(key)
                if embedding:
                    self._embeddings_map[key] = embedding
            
            # Check if eviction needed
            if self._current_size + size_bytes > self.max_size_bytes or len(self._cache) >= self.max_entries:
                await self._evict_intelligent()
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                embedding=embedding,
                hits=0,
                created_at=time.time(),
                last_accessed=time.time(),
                size_bytes=size_bytes,
                quality_score=quality_score
            )
            
            self._cache[key] = entry
            self._current_size += size_bytes
            
            return True
    
    async def _compute_embedding(self, text: str) -> Optional[List[float]]:
        """Compute embedding for semantic search."""
        from jinx.micro.embeddings.embed_cache import embed_text_cached
        
        embedding = await embed_text_cached(text[:512])
        return embedding
    
    async def _find_similar(
        self,
        query: str,
        threshold: float
    ) -> Optional[CacheEntry]:
        """Find semantically similar cached entry."""
        if not self._embeddings_map:
            return None
        
        # Compute query embedding
        query_emb = await self._compute_embedding(query)
        if not query_emb:
            return None
        
        # Find most similar
        best_score = 0.0
        best_key = None
        
        for key, emb in self._embeddings_map.items():
            if key not in self._cache:
                continue
            
            similarity = self._cosine_similarity(query_emb, emb)
            if similarity > best_score and similarity >= threshold:
                best_score = similarity
                best_key = key
        
        if best_key:
            entry = self._cache[best_key]
            entry.hits += 1
            entry.last_accessed = time.time()
            self._cache.move_to_end(best_key)
            return entry
        
        return None
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity."""
        if not a or not b or len(a) != len(b):
            return 0.0
        
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot / (norm_a * norm_b)
    
    async def _evict_intelligent(self) -> None:
        """Intelligent eviction using ML scoring."""
        if not self._cache:
            return
        
        # Score all entries
        scores = []
        for key, entry in self._cache.items():
            # Eviction score = lower is worse
            # Factors: recency, frequency, quality, size
            age = time.time() - entry.last_accessed
            recency_score = 1.0 / (1.0 + age / 3600)  # Decay over hours
            frequency_score = min(1.0, entry.hits / 10.0)
            size_penalty = entry.size_bytes / (1024 * 1024)  # MB
            
            score = (
                recency_score * 0.4 +
                frequency_score * 0.3 +
                entry.quality_score * 0.2 -
                size_penalty * 0.1
            )
            
            scores.append((score, key))
        
        # Sort by score (ascending - worst first)
        scores.sort()
        
        # Evict worst 10% or at least 1
        evict_count = max(1, len(scores) // 10)
        
        for _, key in scores[:evict_count]:
            entry = self._cache.pop(key, None)
            if entry:
                self._current_size -= entry.size_bytes
            
            if key in self._embeddings_map:
                del self._embeddings_map[key]
    
    async def clear(self) -> None:
        """Clear cache."""
        async with self._lock:
            self._cache.clear()
            self._embeddings_map.clear()
            self._current_size = 0
            self._hits = 0
            self._misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'entries': len(self._cache),
            'size_mb': self._current_size / (1024 * 1024),
            'max_size_mb': self.max_size_bytes / (1024 * 1024),
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': hit_rate,
            'embeddings_count': len(self._embeddings_map)
        }


# Singleton
_smart_cache: Optional[SmartCache] = None
_cache_lock = asyncio.Lock()


async def get_smart_cache() -> SmartCache:
    """Get singleton smart cache."""
    global _smart_cache
    if _smart_cache is None:
        async with _cache_lock:
            if _smart_cache is None:
                _smart_cache = SmartCache(max_size_mb=_SMART_CACHE_MB, max_entries=_SMART_CACHE_ENTRIES)
    return _smart_cache


async def cache_get(key: str, **kwargs) -> Optional[Any]:
    """Get from smart cache."""
    cache = await get_smart_cache()
    return await cache.get(key, **kwargs)


async def cache_put(key: str, value: Any, **kwargs) -> bool:
    """Put in smart cache."""
    cache = await get_smart_cache()
    return await cache.put(key, value, **kwargs)


__all__ = [
    "SmartCache",
    "CacheEntry",
    "get_smart_cache",
    "cache_get",
    "cache_put",
]
