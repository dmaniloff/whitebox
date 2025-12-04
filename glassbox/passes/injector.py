"""
Attention injection passes for graph instrumentation.

This module provides inductor passes that intercept and instrument
attention operations in torch FX graphs.
"""

import operator
from abc import abstractmethod
from typing import Callable, List

import torch
from torch import fx
from vllm.compilation.inductor_pass import InductorPass

from ..config import config

VLLM_UNIFIED_ATTENTION_WITH_OUTPUT = "vllm.unified_attention_with_output.default"


class BaseAttentionInjector(InductorPass):
    """
    Base class for attention injectors.

    Provides common functionality for finding attention nodes and writing graphs.

    Args:
        custom_op: A callable (typically a torch.ops operation) to inject.
    """

    def __init__(self, custom_op: Callable):
        super().__init__()
        self.custom_op = custom_op

    def _write_graph_to_file(self, graph: torch.fx.Graph, filename: str) -> None:
        """
        Write the graph to a file in tabular format if tabulate is available.

        Args:
            graph: The FX graph to write
            filename: The output filename
        """
        with open(filename, "w") as f:
            try:
                from tabulate import tabulate

                node_specs = [
                    [n.op, n.name, n.target, n.args, n.kwargs] for n in graph.nodes
                ]
                f.write(
                    tabulate(
                        node_specs,
                        headers=["opcode", "name", "target", "args", "kwargs"],
                    )
                )
            except ImportError:
                f.write(str(graph))

    def _find_attention_nodes(self, graph: torch.fx.Graph) -> List[fx.Node]:
        """
        Find all unified_attention_with_output nodes in the graph.

        Args:
            graph: The FX graph to search

        Returns:
            List of attention nodes found
        """
        nodes = []
        for node in graph.nodes:
            if (
                node.op == "call_function"
                and node.target == torch.ops.higher_order.auto_functionalized
                and node.args
                and str(node.args[0]) == VLLM_UNIFIED_ATTENTION_WITH_OUTPUT
            ):
                nodes.append(node)
        return nodes

    @abstractmethod
    def _inject_instrumentation(
        self, graph: torch.fx.Graph, attention_node: fx.Node
    ) -> None:
        """Inject instrumentation for a single attention node."""
        pass


class PostAttentionInjector(BaseAttentionInjector):
    """
    Injects instrumentation after attention operations.

    Args:
        custom_op: Should accept (tensor, layer_name) and return
                   a tensor with the same shape/dtype/device.
    """

    def __call__(self, graph: torch.fx.Graph) -> None:
        print(f"\n{'=' * 80}")
        print("PostAttentionInjector - Graph Summary")
        print(f"{'=' * 80}")
        print(f"Total nodes: {len(list(graph.nodes))}")
        print(f"{'=' * 80}\n")

        self._write_graph_to_file(graph, config.demo_dir / "graph_before.txt")

        nodes_to_process = self._find_attention_nodes(graph)
        print(f"Found {len(nodes_to_process)} attention nodes to process")

        for attention_node in nodes_to_process:
            self._inject_instrumentation(graph, attention_node)

        self._write_graph_to_file(graph, config.demo_dir / "graph_after.txt")

    def _inject_instrumentation(
        self, graph: torch.fx.Graph, attention_node: fx.Node
    ) -> None:
        """
        Inject instrumentation after an attention node.

        Args:
            graph: The FX graph
            attention_node: The attention node to process
        """
        # The unified_attention_with_output returns a tuple where the second element
        # is the attention output tensor

        layer_name = attention_node.kwargs.get("layer_name", "unknown")

        # Find nodes that use the attention output (second element of the tuple)
        attention_output_users = []

        for user in attention_node.users:
            if (
                user.op == "call_function"
                and user.target == operator.getitem
                and len(user.args) >= 2
                and user.args[1] == 1
            ):  # Getting index 1 (attention output)
                attention_output_users.append(user)

        # For each attention output usage, inject instrumentation
        for att_output_node in attention_output_users:
            with graph.inserting_after(att_output_node):
                # Inject the custom operation
                instrumentation_node = graph.call_function(
                    self.custom_op,
                    args=(att_output_node, layer_name),
                    kwargs={},
                )

                att_output_node.replace_all_uses_with(
                    instrumentation_node,
                    delete_user_cb=lambda n: n is not instrumentation_node,
                )

                print(
                    f"Injected instrumentation after attention output at {att_output_node.name}"
                )


def create_post_attention_injector(custom_op: Callable):
    """
    Factory function to create a PostAttentionInjector instance.

    Args:
        custom_op: A callable (typically a torch.ops operation) to inject after
                   attention outputs. Should accept (tensor, layer_name) and return
                   a tensor with the same shape/dtype/device as the input.

    Returns:
        A configured PostAttentionInjector instance.
    """
    return PostAttentionInjector(custom_op)


class BeforeAttentionInjector(BaseAttentionInjector):
    """
    Injects instrumentation before attention operations with access to Q, K, V.

    Args:
        custom_op: Should accept (q, k, v, layer_name) and return
                   a tuple of tensors (q, k, v) with the same shapes/dtypes/devices.
    """

    def __call__(self, graph: torch.fx.Graph) -> None:
        print(f"\n{'=' * 80}")
        print("BeforeAttentionInjector - Graph Summary")
        print(f"{'=' * 80}")
        print(f"Total nodes: {len(list(graph.nodes))}")
        print(f"{'=' * 80}\n")

        self._write_graph_to_file(graph, config.demo_dir / "graph_before_qkv.txt")

        nodes_to_process = self._find_attention_nodes(graph)
        print(f"Found {len(nodes_to_process)} attention nodes to process")

        for attention_node in nodes_to_process:
            self._inject_instrumentation(graph, attention_node)

        self._write_graph_to_file(graph, config.demo_dir / "graph_after_qkv.txt")

    def _inject_instrumentation(
        self, graph: torch.fx.Graph, attention_node: fx.Node
    ) -> None:
        """
        Inject instrumentation before an attention node, intercepting Q, K, V.

        Args:
            graph: The FX graph
            attention_node: The attention node to process
        """
        layer_name = attention_node.kwargs.get("layer_name", "unknown")

        # Get the Q, K, V nodes from the attention kwargs
        q_node = attention_node.kwargs.get("query")
        k_node = attention_node.kwargs.get("key")
        v_node = attention_node.kwargs.get("value")

        if q_node is None or k_node is None or v_node is None:
            print(
                f"Warning: Could not find Q, K, V for attention node at {attention_node.name}"
            )
            return

        # Insert the custom op before the attention node
        with graph.inserting_before(attention_node):
            # Call the custom op with q, k, v
            instrumentation_node = graph.call_function(
                self.custom_op,
                args=(q_node, k_node, v_node, layer_name),
                kwargs={},
            )

            # Extract the individual q, k, v outputs from the tuple
            new_q = graph.call_function(
                operator.getitem, args=(instrumentation_node, 0)
            )
            new_k = graph.call_function(
                operator.getitem, args=(instrumentation_node, 1)
            )
            new_v = graph.call_function(
                operator.getitem, args=(instrumentation_node, 2)
            )

        # Update the attention node to use the new q, k, v
        new_kwargs = dict(attention_node.kwargs)
        new_kwargs["query"] = new_q
        new_kwargs["key"] = new_k
        new_kwargs["value"] = new_v
        attention_node.kwargs = new_kwargs

        print(f"Injected QKV instrumentation before attention at {attention_node.name}")


def create_before_attention_injector(custom_op: Callable):
    """
    Factory function to create a BeforeAttentionInjector instance.

    Args:
        custom_op: A callable (typically a torch.ops operation) to inject before
                   attention. Should accept (q, k, v, layer_name) and return
                   a tuple of tensors (q, k, v) with the same shapes/dtypes/devices.

    Returns:
        A configured BeforeAttentionInjector instance.
    """
    return BeforeAttentionInjector(custom_op)
