from typing import Any, Dict, List

import numpy as np
from numpy.typing import NDArray

class Node:
    """Node in a computational graph.

    Fields
    ------
    inputs: List[Node]
        The list of input nodes to this node.

    op: Op
        The op of this node.

    attrs: Dict[str, Any]
        The attribute dictionary of this node.
        E.g. "constant" is the constant operand of add_by_const.

    name: str
        Name of the node for debugging purposes.
    """

    inputs: List["Node"]
    op: "Op"
    attrs: Dict[str, Any]
    name: str

    def __init__(
        self, inputs: List["Node"], op: "Op", attrs: Dict[str, Any] = {}, name: str = ""
    ) -> None:
        self.inputs = inputs
        self.op = op
        self.attrs = attrs
        self.name = name

    def __add__(self, other):
        if isinstance(other, Node):
            return add(self, other)
        else:
            assert isinstance(other, (int, float))
            return add_by_const(self, other)

    def __sub__(self, other):
        return self + (-1) * other

    def __rsub__(self, other):
        return (-1) * self + other

    def __mul__(self, other):
        if isinstance(other, Node):
            return mul(self, other)
        else:
            assert isinstance(other, (int, float))
            return mul_by_const(self, other)

    def __truediv__(self, other):
        if isinstance(other, Node):
            return div(self, other)
        else:
            assert isinstance(other, (int, float))
            return div_by_const(self, other)

    # Allow left-hand-side add and multiplication.
    __radd__ = __add__
    __rmul__ = __mul__

    def __str__(self):
        """Allow printing the node name."""
        return self.name

    def __getattr__(self, attr_name: str) -> Any:
        if attr_name in self.attrs:
            return self.attrs[attr_name]
        raise KeyError(f"Attribute {attr_name} does not exist in node {self}")

    __repr__ = __str__


class Variable(Node):
    """A variable node with given name."""

    def __init__(self, name: str) -> None:
        super().__init__(inputs=[], op=placeholder, name=name)


class Op:
    """The class of operations performed on nodes."""

    def __call__(self, *kwargs) -> Node:
        """Create a new node with this current op.

        Returns
        -------
        The created new node.
        """
        raise NotImplementedError

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Compute the output value of the given node with its input
        node values given.

        Parameters
        ----------
        node: Node
            The node whose value is to be computed

        input_values: List[np.ndarray]
            The input values of the given node.

        Returns
        -------
        output: np.ndarray
            The computed output value of the node.
        """
        raise NotImplementedError

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given a node and its output gradient node, compute partial
        adjoints with regards to each input node.

        Parameters
        ----------
        node: Node
            The node whose inputs' partial adjoints are to be computed.

        output_grad: Node
            The output gradient with regard to given node.

        Returns
        -------
        input_grads: List[Node]
            The list of partial gradients with regard to each input of the node.
        """
        raise NotImplementedError


class PlaceholderOp(Op):
    """The placeholder op to denote computational graph input nodes."""

    def __call__(self, name: str) -> Node:
        return Node(inputs=[], op=self, name=name)

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        raise RuntimeError(
            "Placeholder nodes have no inputs, and there values cannot be computed."
        )

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        raise RuntimeError("Placeholder nodes have no inputs.")


class AddOp(Op):
    """Op to element-wise add two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}+{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return the element-wise addition of input values."""
        assert len(input_values) == 2
        return input_values[0] + input_values[1]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of add node, return partial adjoint to each input."""
        return [output_grad, output_grad]


class AddByConstOp(Op):
    """Op to element-wise add a node by a constant."""

    def __call__(self, node_A: Node, const_val: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"constant": const_val},
            name=f"({node_A.name}+{const_val})",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return the element-wise addition of the input value and the constant."""
        assert len(input_values) == 1
        return input_values[0] + node.constant

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of add node, return partial adjoint to the input."""
        return [output_grad]


class MulOp(Op):
    """Op to element-wise multiply two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}*{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return the element-wise multiplication of input values."""
        assert len(input_values) == 2
        return input_values[0] * input_values[1]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of multiplication node, return partial adjoint to each input."""
        return[output_grad*node.inputs[1], output_grad*node.inputs[0]]


class MulByConstOp(Op):
    """Op to element-wise multiply a node by a constant."""

    def __call__(self, node_A: Node, const_val: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"constant": const_val},
            name=f"({node_A.name}*{const_val})",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return the element-wise multiplication of the input value and the constant."""
        assert len(input_values) == 1
        return input_values[0] * node.constant

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of multiplication node, return partial adjoint to the input."""
        return[output_grad*node.constant]


class DivOp(Op):
    """Op to element-wise divide two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}/{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return the element-wise division of input values."""
        assert len(input_values) == 2
        return input_values[0] / input_values[1]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of division node, return partial adjoint to each input."""
        return[output_grad/node.inputs[1], output_grad*(-1*node.inputs[0])/(node.inputs[1]*node.inputs[1])]


class DivByConstOp(Op):
    """Op to element-wise divide a nodes by a constant."""

    def __call__(self, node_A: Node, const_val: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"constant": const_val},
            name=f"({node_A.name}/{const_val})",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return the element-wise division of the input value and the constant."""
        assert len(input_values) == 1
        return input_values[0] / node.constant

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of division node, return partial adjoint to the input."""
        return [output_grad/node.constant]


class MatMulOp(Op):
    """Matrix multiplication op of two nodes."""

    def __call__(
        self, node_A: Node, node_B: Node, trans_A: bool = False, trans_B: bool = False
    ) -> Node:
        """Create a matrix multiplication node.

        Parameters
        ----------
        node_A: Node
            The lhs matrix.
        node_B: Node
            The rhs matrix
        trans_A: bool
            A boolean flag denoting whether to transpose A before multiplication.
        trans_B: bool
            A boolean flag denoting whether to transpose B before multiplication.

        Returns
        -------
        result: Node
            The node of the matrix multiplication.
        """
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={"trans_A": trans_A, "trans_B": trans_B},
            name=f"({node_A.name + ('.T' if trans_A else '')}@{node_B.name + ('.T' if trans_B else '')})",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return the matrix multiplication result of input values.

        Note
        ----
        For this assignment, you can assume the matmul only works for 2d matrices.
        That being said, the test cases guarantee that input values are
        always 2d numpy.ndarray.
        """
        assert len(input_values) == 2
        if node.trans_A:
            input_values[0] = input_values[0].T
        if node.trans_B:
            input_values[1] = input_values[1].T
        out = input_values[0] @ input_values[1]
        node.attrs["shape"] = out.shape
        return out
            

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of matmul node, return partial adjoint to each input.

        Note
        ----
        - Same as the `compute` method, you can assume that the input are 2d matrices.
        However, it would be a good exercise to think about how to handle
        more general cases, i.e., when input can be either 1d vectors,
        2d matrices, or multi-dim tensors.
        - You may want to look up some materials for the gradients of matmul.
        """
        A, B = node.inputs
        trans_A = node.trans_A
        trans_B = node.trans_B
        
        if not trans_A and not trans_B:
            return [MatMulOp()(output_grad, B, trans_B=True), MatMulOp()(A,output_grad,trans_A=True)]
        elif trans_A and not trans_B:
            return [MatMulOp()(B, output_grad, trans_B=True), MatMulOp()(A, output_grad)]
        elif not trans_A and trans_B:
            return [MatMulOp()(output_grad, B),MatMulOp()(output_grad, A, trans_A=True)]
        else:  # trans_A and trans_B
            return [MatMulOp()(B, output_grad, trans_A=True,trans_B=True), MatMulOp()(output_grad, A, trans_A=True,trans_B=True)]

class ZerosLikeOp(Op):
    """Zeros-like op that returns an all-zero array with the same shape as the input."""

    def __call__(self, node_A: Node) -> Node:
        return Node(inputs=[node_A], op=self, name=f"ZerosLike({node_A.name})")

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return an all-zero tensor with the same shape as input."""
        assert len(input_values) == 1
        return np.zeros(input_values[0].shape)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [zeros_like(node.inputs[0])]


class OnesLikeOp(Op):
    """Ones-like op that returns an all-one array with the same shape as the input."""

    def __call__(self, node_A: Node) -> Node:
        return Node(inputs=[node_A], op=self, name=f"OnesLike({node_A.name})")

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return an all-one tensor with the same shape as input."""
        assert len(input_values) == 1
        return np.ones(input_values[0].shape)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [zeros_like(node.inputs[0])]

#new OP 
class ExpOp(Op):
    def __call__(self, node_A: Node) -> Node:
        return Node(inputs=[node_A], op=self, name=f"exp({node_A.name})")

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
            x = input_values[0].astype(np.float64)
            return np.exp(x)


    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [output_grad * exp(node.inputs[0])]

class SumOp(Op):
    def __call__(self, node_A: Node, axis: Any = None, keepdims: bool = False) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"axis": axis, "keepdims": keepdims},
            name=f"sum({node_A.name},axis={axis},keepdims={keepdims})"
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        return np.sum(input_values[0], axis=node.axis, keepdims=node.keepdims)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [output_grad * ones_like(node.inputs[0])]

class LogOp(Op):
    def __call__(self, node_A: Node) -> Node:
        return Node(inputs=[node_A], op=self, name=f"log({node_A.name})")

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        return np.log(input_values[0])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [output_grad / node.inputs[0]]


class BroadcastLikeOp(Op):
    def __call__(self, b: Node, ref: Node) -> Node:
        return Node(inputs=[b, ref],
                    op=self,
                    name=f"broadcast({b.name} -> {ref.name})")
    
    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        b_val, ref_val = input_values
        return np.broadcast_to(b_val, ref_val.shape)

    def gradient(self, node: Node, out_grad: Node) -> List[Node]:
        grad_b = sum(out_grad, axis=0)  
        grad_ref = zeros_like(node.inputs[1])
        return [grad_b, grad_ref]


# Create global instances of ops.
# Your implementation should just use these instances, rather than creating new instances.
placeholder = PlaceholderOp()
add = AddOp()
mul = MulOp()
div = DivOp()
add_by_const = AddByConstOp()
mul_by_const = MulByConstOp()
div_by_const = DivByConstOp()
matmul = MatMulOp()
zeros_like = ZerosLikeOp()
ones_like = OnesLikeOp()
exp = ExpOp()
log = LogOp()
sum = SumOp()
broadcast_to_like = BroadcastLikeOp()


class Evaluator:
    """The node evaluator that computes the values of nodes in a computational graph."""

    eval_nodes: List[Node]

    def __init__(self, eval_nodes: List[Node]) -> None:
        """Constructor, which takes the list of nodes to evaluate in the computational graph.

        Parameters
        ----------
        eval_nodes: List[Node]
            The list of nodes whose values are to be computed.
        """
        self.eval_nodes = eval_nodes

    def run(self, input_values: Dict[Node, np.ndarray]) -> List[np.ndarray]:
        """Computes values of nodes in `eval_nodes` field with
        the computational graph input values given by the `input_values` dict.

        Parameters
        ----------
        input_values: Dict[Node, np.ndarray]
            The dictionary providing the values for input nodes of the
            computational graph.
            Throw ValueError when the value of any needed input node is
            not given in the dictionary.

        Returns
        -------
        eval_values: List[np.ndarray]
            The list of values for nodes in `eval_nodes` field.
        """
        computed_values = {}
        visited = set()
        sorted_nodes = []

        def topo_sort(node, visited, sorted_nodes):
            if node in visited:
                return
            visited.add(node)
            for input_node in getattr(node, 'inputs' , []):
                topo_sort(input_node, visited, sorted_nodes)
            sorted_nodes.append(node)
        
        for node in self.eval_nodes:
            topo_sort(node, visited, sorted_nodes)
        
        for node in sorted_nodes:
            if node in input_values:
                computed_values[node] = input_values[node]
            else:
                input_vals = [computed_values[input_node] for input_node in node.inputs]
                if node.op is None:
                    raise ValueError(f"Node {node} has no operation and is not in input_values.")
                computed_values[node] = node.op.compute(node, input_vals)
        return [computed_values[node] for node in self.eval_nodes]



def gradients(output_node: Node, nodes: List[Node]) -> List[Node]:
    """Construct the backward computational graph, which takes gradient
    of given output node with respect to each node in input list.
    Return the list of gradient nodes, one for each node in the input list.

    Parameters
    ----------
    output_node: Node
        The output node to take gradient of, whose gradient is 1.

    nodes: List[Node]
        The list of nodes to take gradient with regard to.

    Returns
    -------
    grad_nodes: List[Node]
        A list of gradient nodes, one for each input nodes respectively.
    """

    grad_map = {}
    reverse_sorted_nodes = []
    visited = set()

    def topo_sort(node: Node):
        if node in visited:
            return
        visited.add(node)
        for input_node in node.inputs:
            topo_sort(input_node)
        reverse_sorted_nodes.append(node)
    
    topo_sort(output_node)
    reverse_sorted_nodes.reverse()

    grad_map[output_node] = ones_like(output_node)

    print(reverse_sorted_nodes)

    for node in reverse_sorted_nodes:
        if node not in grad_map:
            continue

        if node.op is None or isinstance(node.op, PlaceholderOp):
            continue
        current_grad = grad_map[node]

        input_grads = node.op.gradient(node, current_grad)
        for input_node, grad in zip(node.inputs, input_grads):
            if input_node not in grad_map:
                grad_map[input_node] = grad
            else:
                grad_map[input_node] = grad_map[input_node] + grad
    return [grad_map.get(node, zeros_like(node)) for node in nodes]
