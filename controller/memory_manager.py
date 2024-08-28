import torch
from typing import List, Dict, Optional
import logging


class MemoryManager:
    def __init__(self, model, max_memories: int = 5):
        self.model = model
        self.max_memories = max_memories
        self.memories: List[Dict] = []
        self.logger = logging.getLogger(__name__)

    def add_memory(
        self, crv: torch.Tensor, seq_length: int, layer_idx: int, crv_layers: List[int]
    ):
        """
        Add a new memory (CRV) to the manager.
        """
        if len(self.memories) >= self.max_memories:
            self.memories.pop(0)  # Remove the oldest memory if we've reached the limit

        memory = {
            "crv": crv[:, :seq_length, :],  # Store only up to seq_length
            "seq_length": seq_length,
            "layer_idx": layer_idx,
            "crv_layers": crv_layers,
            "start_pos": None,
            "end_pos": None,
        }
        self.memories.append(memory)
        self.logger.info(
            f"Added new memory. Current number of memories: {len(self.memories)}"
        )

    def set_concat_positions(self, memory_idx: int, start_pos: int, end_pos: int):
        """
        Set the start and end positions for concatenation in the hidden state.
        """
        if 0 <= memory_idx < len(self.memories):
            self.memories[memory_idx]["start_pos"] = start_pos
            self.memories[memory_idx]["end_pos"] = end_pos
            self.logger.info(
                f"Set concat positions for memory {memory_idx}: start={start_pos}, end={end_pos}"
            )
        else:
            self.logger.warning(f"Invalid memory index: {memory_idx}")

    def get_memory(self, idx: int) -> Optional[Dict]:
        """
        Retrieve a specific memory by index.
        """
        if 0 <= idx < len(self.memories):
            return self.memories[idx]
        else:
            self.logger.warning(f"Invalid memory index: {idx}")
            return None

    def clear_memories(self):
        """
        Clear all stored memories.
        """
        self.memories.clear()
        self.logger.info("Cleared all memories")

    def apply_memory_to_model(self, memory_idx: int):
        """
        Apply a specific memory to the model.
        """
        memory = self.get_memory(memory_idx)
        if memory:
            self.model.set_crv(
                memory["crv"],
                memory["layer_idx"],
                memory["crv_layers"],
                start_pos=memory["start_pos"],
                end_pos=memory["end_pos"],
            )
            self.logger.info(f"Applied memory {memory_idx} to model")
        else:
            self.logger.warning(f"Failed to apply memory {memory_idx}")

    def get_total_memories(self) -> int:
        """
        Get the total number of stored memories.
        """
        return len(self.memories)

    def get_memory_stats(self) -> Dict:
        """
        Get statistics about the stored memories.
        """
        stats = {
            "total_memories": len(self.memories),
            "max_seq_length": (
                max(m["seq_length"] for m in self.memories) if self.memories else 0
            ),
            "min_seq_length": (
                min(m["seq_length"] for m in self.memories) if self.memories else 0
            ),
            "avg_seq_length": (
                sum(m["seq_length"] for m in self.memories) / len(self.memories)
                if self.memories
                else 0
            ),
        }
        self.logger.info(f"Memory stats: {stats}")
        return stats

    def update_memory(self, idx: int, **kwargs):
        """
        Update specific fields of a memory.
        """
        if 0 <= idx < len(self.memories):
            self.memories[idx].update(kwargs)
            self.logger.info(f"Updated memory {idx} with {kwargs}")
        else:
            self.logger.warning(f"Invalid memory index: {idx}")
