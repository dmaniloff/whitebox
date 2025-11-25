import operator

import torch
from torch import fx
from vllm.compilation.inductor_pass import InductorPass

from ..config import config

VLLM_UNIFIED_ATTENTION_WITH_OUTPUT = "vllm.unified_attention_with_output.default"


# Register a custom torch operation
@torch.library.custom_op("glassbox::capture_mean", mutates_args=())
def capture_mean_op(x: torch.Tensor, layer_name: str) -> torch.Tensor:
    """Custom op to capture mean values."""
    mean_val = float(x.mean().item())
    print(f"[MEAN_CAPTURE] {layer_name} attention mean: {mean_val:.6f}")
    return x.clone()  # Passthrough


# Register the fake/abstract implementation for torch.compile
@capture_mean_op.register_fake
def _(x: torch.Tensor, layer_name: str) -> torch.Tensor:
    return x.clone()  # "Output has same shape/dtype/device as input"


class AttentionMeanPass(InductorPass):
    """
    A custom inductor pass that calculates the mean of attention outputs.

    This pass finds all unified_attention_with_output operations in the graph
    and injects a mean calculation after each one.
    """

    def __init__(self):
        super().__init__()

    def _write_graph_to_file(self, graph: torch.fx.Graph, filename: str) -> None:
        """
        Write the graph to a file in tabular format if tabulate is available.

        Args:
            graph: The FX graph to write
            filename: The output filename
        """
        with open(filename, "w") as f:
            try:
                # mimic graph.print_tabular but print to a file
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
                # Fallback to string representation
                f.write(str(graph))

    def __call__(self, graph: torch.fx.Graph) -> None:
        # Print a summary of the graph
        print(f"\n{'=' * 80}")
        print("Graph Summary")
        print(f"{'=' * 80}")
        print(f"Total nodes: {len(list(graph.nodes))}")
        print(f"{'=' * 80}\n")

        self._write_graph_to_file(graph, config.demo_dir / "graph_before.txt")

        # Keep track of nodes to avoid modifying graph while iterating
        nodes_to_process = []

        # Find all unified_attention_with_output nodes
        # How model-specific is this?
        for node in graph.nodes:
            if (
                node.op == "call_function"
                and node.target == torch.ops.higher_order.auto_functionalized
            ):
                # Check if it's the attention operation
                if (
                    node.args
                    and str(node.args[0]) == VLLM_UNIFIED_ATTENTION_WITH_OUTPUT
                ):
                    nodes_to_process.append(node)

        print(f"Found {len(nodes_to_process)} attention nodes to process")

        # Process each attention node
        for attention_node in nodes_to_process:
            self._inject_mean_calculation(graph, attention_node)

        self._write_graph_to_file(graph, config.demo_dir / "graph_after.txt")

    def _inject_mean_calculation(
        self, graph: torch.fx.Graph, attention_node: fx.Node
    ) -> None:
        """
        Inject mean calculation after an attention node.

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

        # For each attention output usage, inject a mean calculation
        for att_output_node in attention_output_users:
            with graph.inserting_after(att_output_node):
                # Calculate mean via a capture node
                capture_node = graph.call_function(
                    torch.ops.glassbox.capture_mean.default,
                    args=(att_output_node, layer_name),
                    kwargs={},
                )

                att_output_node.replace_all_uses_with(
                    capture_node, delete_user_cb=lambda n: n is not capture_node
                )

                print(
                    f"Injected mean calculation after attention output at {att_output_node.name}"
                )


def create_attention_mean_pass():
    return AttentionMeanPass()
