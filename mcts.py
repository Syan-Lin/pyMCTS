from abc import ABC, abstractmethod
from time import time
from graphviz import Digraph
import math

class MonteCarloTreeNode(ABC):
    exploration_weight: float = 1.0

    def __init__(self, minmax_search: bool = True):
        self.visit = 0
        self.score = 0
        self.children = []
        self.is_terminal = False
        self.parent: MonteCarloTreeNode = None
        self.minmax_search = minmax_search

    @abstractmethod
    def expand(self) -> list["MonteCarloTreeNode"]:
        raise NotImplementedError("Expand method must be implemented by subclasses.")

    @abstractmethod
    def simulate(self) -> float:
        raise NotImplementedError("Simulate method must be implemented by subclasses.")

    def expand_node(self) -> None:
        self.children = self.expand()
        if len(self.children) == 0:
            self.is_terminal = True
            return

        for child in self.children:
            child.parent = self

    def backpropagate(self, result: float) -> None:
        self.visit += 1
        self.score += result
        if self.parent:
            self.parent.backpropagate(-result if self.minmax_search else result)

    def select(self, N: int) -> "MonteCarloTreeNode":
        if len(self.children) == 0:
            return self

        best_child = self.children[0]
        best_ucb1 = best_child.ucb1(N)

        for child in self.children:
            if child.visit == 0:
                return child
            ucb1 = child.ucb1(N)
            if ucb1 > best_ucb1:
                best_ucb1 = ucb1
                best_child = child

        return best_child

    def ucb1(self, N: int) -> float:
        if self.visit == 0:
            return float('inf')
        return (self.score / self.visit) + MonteCarloTreeNode.exploration_weight * (math.log(N) / self.visit)

class MonteCarloTree(ABC):
    def __init__(self, root: MonteCarloTreeNode):
        self.root = root

    def search(self, iterations: int = None, time_limit: float = None) -> MonteCarloTreeNode:
        # This method performs the MCTS search, returns the best node after the search is complete.
        if iterations is None and time_limit is None:
            raise ValueError("Either iterations or time_limit must be specified.")
        if iterations is not None and time_limit is not None:
            raise ValueError("Only one of iterations or time_limit can be specified.")
        if time_limit and time_limit <= 0:
            raise ValueError("Time limit must be greater than zero.")
        if iterations and iterations <= 0:
            raise ValueError("Number of iterations must be greater than zero.")

        def search_iter(curr: MonteCarloTreeNode) -> MonteCarloTreeNode:
            while len(curr.children) > 0:
                curr = curr.select(self.root.visit)
            if not curr.is_terminal and curr.visit != 0:
                curr.expand_node()
                if not curr.is_terminal:
                    curr = curr.children[0]
            score = curr.simulate()
            curr.backpropagate(score)

            return curr

        curr = self.root
        if iterations is not None:
            for _ in range(iterations):
                search_iter(curr)
        else:
            CHECK_ITERATIONS = 100
            iters = 0
            start_time = time()
            while iters % CHECK_ITERATIONS != 0 and time() - start_time < time_limit:
                search_iter(curr)
                iters += 1

        # Find the best child based on visit count
        if len(self.root.children) == 0:
            raise ValueError("No children found in the root node. Ensure the root could be expanded.")
        best_child = self.root.children[0]
        for child in self.root.children[1:]:
            if child.visit > best_child.visit:
                best_child = child

        return best_child

    def move_to(self, node: MonteCarloTreeNode) -> None:
        # This method moves the root to the specified node.
        if not isinstance(node, MonteCarloTreeNode):
            raise TypeError("node must be an instance of MonteCarloTreeNode.")
        if node is None:
            raise ValueError("node cannot be None.")

        self.root = node
        self.root.parent = None

class TreeDrawer():
    def __init__(self):
        self.frame = 0

    def draw(self, tree, filename: str = "mcts_tree") -> None:
        dot = Digraph(comment='Monte Carlo Tree')
        dot.attr('node', shape='box', style='filled', fillcolor='lightblue')
        dot.attr('edge', fontsize='10')

        # Mark the select route
        curr = tree.root
        select_route = []
        while len(curr.children) > 0:
            curr = curr.select(tree.root.visit)
            select_route.append(curr)

        # Add nodes recursively
        def add_nodes(node: MonteCarloTreeNode) -> None:
            label = f"Visits: {node.visit}\nScore: {node.score:.2f}"
            if node.visit > 0:
                label += f"\nAvg: {node.score/node.visit:.2f}"

            node_id = str(id(node))
            dot.node(node_id, label)

            if node is tree.root:
                # Root node style
                dot.node(node_id, label, style='filled', fillcolor='gold')
            elif node.is_terminal:
                # Terminal node style
                dot.node(node_id, label, style='filled', fillcolor='salmon')
            elif node in select_route:
                # Selected node style
                dot.node(node_id, label, style='filled', fillcolor='lightgreen')

            for child in node.children:
                child_id = str(id(child))
                ucb1 = "INF" if child.visit == 0 else f"{child.ucb1(node.visit):.4f}"
                dot.edge(node_id, child_id, label=ucb1)
                add_nodes(child)
        add_nodes(tree.root)

        dot.graph_attr['dpi'] = str(300)
        dot.format = 'pdf'
        dot.render(f"{filename}_{self.frame}", view=False, cleanup=True)
        print(f"Save {filename}_{self.frame}.pdf")
        self.frame += 1