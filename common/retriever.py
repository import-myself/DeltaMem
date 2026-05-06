"""
Vector Retriever
向量检索器：负责文本向量化与高效相似度计算
"""

import numpy as np
import logging
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL_PATH, 
    SIMILARITY_THRESHOLD, 
    TOP_K_CANDIDATES, 
)
from memory.prtree.memory_node import MemoryNode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorRetriever:
    """
    向量检索器
    
    职责:
    1. Text -> Vector (Encoding)
    2. Vector vs Matrix (Batch Similarity Calculation)
    3. Threshold Filtering
    """
    
    def __init__(self, model_path: str = EMBEDDING_MODEL_PATH):
        logger.info(f"Loading embedding model from: {model_path}")
        try:
            self.model = SentenceTransformer(model_path)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.warning("Falling back to MOCK embeddings (Random Noise) for testing.")
            self.model = None
            self.embedding_dim = 384 
    
    def encode(self, text: str) -> np.ndarray:
        """将文本转换为向量"""
        if not text:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        if self.model is None:
            np.random.seed(abs(hash(text)) % (2**32))
            vec = np.random.randn(self.embedding_dim).astype(np.float32)
            return vec / np.linalg.norm(vec)
        
        embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding.astype(np.float32)
    
    def _batch_cosine_similarity(self, query_emb: np.ndarray, candidates_emb: np.ndarray) -> np.ndarray:
        """批量计算 Cosine Similarity"""
        q_norm = np.linalg.norm(query_emb)
        c_norms = np.linalg.norm(candidates_emb, axis=1)
        
        if q_norm == 0:
            return np.zeros(len(candidates_emb))
            
        dot_products = np.dot(candidates_emb, query_emb)
        scores = dot_products / (q_norm * (c_norms + 1e-10))
        return scores

    def retrieve_candidates(
        self, 
        query_embedding: np.ndarray, 
        candidate_nodes: List[MemoryNode],
        threshold: float = SIMILARITY_THRESHOLD,
        top_k: Optional[int] = TOP_K_CANDIDATES
    ) -> List[Tuple[MemoryNode, float]]:
        """
        核心检索函数
        
        Args:
            top_k: 如果为 None，则返回所有满足阈值的节点 (用于 BFS)
        """
        if not candidate_nodes:
            return []
        
        # 1. 准备矩阵
        candidate_matrix = np.array([node.embedding for node in candidate_nodes])
        
        # 2. 批量计算
        scores = self._batch_cosine_similarity(query_embedding, candidate_matrix)
        
        # 3. 过滤与打包
        results = []
        for i, score in enumerate(scores):
            if score >= threshold:
                results.append((candidate_nodes[i], float(score)))
        
        # 4. 排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        # 5. 截断 (如果指定了 top_k)
        if top_k is not None:
            return results[:top_k]
        
        return results

    def retrieve_all_matches(
        self,
        query_embedding: np.ndarray,
        candidate_nodes: List[MemoryNode],
        threshold: float = SIMILARITY_THRESHOLD
    ) -> List[Tuple[MemoryNode, float]]:
        """
        获取所有匹配节点（专门用于 BFS/Beam Search）
        语义上等同于 retrieve_candidates(..., top_k=None)
        """
        return self.retrieve_candidates(
            query_embedding=query_embedding,
            candidate_nodes=candidate_nodes,
            threshold=threshold,
            top_k=None  # Explicitly return all
        )
    
    def retrieve_best_match(
        self, 
        query_embedding: np.ndarray, 
        candidate_nodes: List[MemoryNode],
        threshold: float = SIMILARITY_THRESHOLD
    ) -> Optional[Tuple[MemoryNode, float]]:
        """获取最佳单点 (DFS 贪婪搜索用)"""
        results = self.retrieve_candidates(query_embedding, candidate_nodes, threshold, top_k=1)
        if results:
            return results[0]
        return None