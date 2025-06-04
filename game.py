from mcts import MonteCarloTree, MonteCarloTreeNode
import numpy as np
from graphviz import Digraph

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
            label += '\n' + str(node)

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

class TicTacToeNode(MonteCarloTreeNode):
    SYMBOLS = {0: 'X', 1: 'O', None: '.'}

    def __init__(self, current_player: int, board: np.ndarray = None):
        super().__init__(current_player)
        self.board = board if board is not None else np.full((3, 3), None)
        self.current_player = current_player

        self.winner = self.check_winner(self.board)
        if (self.winner is not None) or (not None in self.board):
            self.is_terminal = True

    def check_winner(self, board) -> int:
        # Check if the game is over and return the winner, None if no winner
        for i in range(3):
            if board[i, 0] == board[i, 1] == board[i, 2] is not None:
                return board[i, 0]

        for j in range(3):
            if board[0, j] == board[1, j] == board[2, j] is not None:
                return board[0, j]

        if board[0, 0] == board[1, 1] == board[2, 2] is not None:
            return board[0, 0]
        if board[0, 2] == board[1, 1] == board[2, 0] is not None:
            return board[0, 2]

        return None

    def expand(self) -> list["TicTacToeNode"]:
        children = []
        for i in range(3):
            for j in range(3):
                if self.board[i, j] is None:
                    new_board = self.board.copy()
                    new_board[i, j] = self.current_player
                    child = TicTacToeNode(1 - self.current_player, new_board)
                    children.append(child)
        return children

    def simulate(self) -> float:
        board = self.board.copy()
        current_player = self.current_player

        # The game is draw
        winner = self.check_winner(board)
        if winner is None and None not in board:
            return 0

        # Random play
        while True:
            # Find all empty positions
            empty_positions = []
            for i in range(3):
                for j in range(3):
                    if board[i, j] is None:
                        empty_positions.append((i, j))
            if not empty_positions:
                break

            # Randomly select a position
            i, j = empty_positions[np.random.randint(len(empty_positions))]
            board[i, j] = current_player
            current_player = 1 - current_player

            # Check for victory
            winner = self.check_winner(board)
            if winner is not None:
                break

        if winner is None:
            return 0
        elif winner == current_player:
            return 1
        else:
            return -1

    def __str__(self):
        s = ""
        for i in range(3):
            for j in range(3):
                s += TicTacToeNode.SYMBOLS[self.board[i, j]] + " "
            s += "\n"
        return s

class TicTacToeGame:
    slience = False

    def __init__(self, player_first: str, player_second: str, iterations: int):
        self.round = 0
        self.player_first = player_first
        self.player_second = player_second
        self.iterations = iterations
        self.drawer = TreeDrawer()
        self.reset()

    def reset(self):
        self.root = TicTacToeNode(0)
        self.tree = MonteCarloTree(self.root)
        self.current_node = self.root

    def display_board(self, player: str):
        if TicTacToeGame.slience:
            return
        print(f"Current board (Round {self.round}, {player} move):")
        print(self.current_node)
        winner = self.current_node.check_winner(self.current_node.board)
        if winner is None and None not in self.current_node.board:
            print("Game over! Draw!")
        elif winner == 0:
            print("X wins!")
        elif winner == 1:
            print("O wins!")
        self.round += 1

    def human_move(self):
        while True:
            move = input("Input: ").split()
            if len(move) != 2:
                print("Invalid input! Please enter row and column as two numbers (1-3).")
                continue

            i, j = int(move[0]), int(move[1])
            if not (1 <= i <= 3 and 1 <= j <= 3):
                print("Invalid move! Please enter row and column as two numbers (1-3).")
                continue

            if self.current_node.board[i - 1, j - 1] is not None:
                print("This position is already occupied!")
                continue

            new_board = self.current_node.board.copy()
            new_board[i - 1, j - 1] = self.current_node.current_player
            new_node = TicTacToeNode(1 - self.current_node.current_player, new_board)
            return new_node

    def ai_move(self):
        best_child = self.tree.search(iterations=self.iterations)

        # Visualize the tree
        # self.drawer.draw(self.tree)

        return best_child

    def play(self):
        cnt = 0

        while not self.current_node.is_terminal:
            if cnt % 2 == 0:
                curr = self.player_first
            else:
                curr = self.player_second
            cnt += 1


            if curr == "human":
                new_node = self.human_move()
            else:
                new_node = self.ai_move()

            # Update game state
            self.current_node = new_node
            self.tree.move_to(self.current_node)
            self.display_board(curr)

if __name__ == "__main__":
    # Select game mode
    print("Select game mode:")
    print("1: Human vs AI")
    print("2: AI vs AI")
    print("3: AI vs AI (Batch)")

    mode = input("Select (1-3): ")
    modes = {
        '1': ('human', 'ai'),
        '2': ('ai', 'ai'),
        '3': ('ai', 'ai')
    }

    if mode not in modes:
        print("Invalid selection, defaulting to Human vs AI")
        mode = '1'

    player1, player2 = modes[mode]

    # Set MCTS iteration count
    iterations = 1000
    if player1 == 'ai' or player2 == 'ai':
        try:
            iterations = int(input("Set AI search iteration count (default 1000): ") or 1000)
        except ValueError:
            iterations = 1000

    # Start game
    if mode == '3':
        TicTacToeGame.slience = True
        print("Set batch size:")
        batch_size = int(input("Batch size (default 10): ") or 10)
        wins, draws, losses = 0, 0, 0
        for i in range(batch_size):
            game = TicTacToeGame(player1, player2, iterations)
            game.play()
            winner = game.current_node.check_winner(game.current_node.board)
            if winner is None:
                draws += 1
            elif winner == 0:
                wins += 1
            else:
                losses += 1
        print(f"Batch results(first hand): Wins: {wins}, Draws: {draws}, Losses: {losses}")
    else:
        game = TicTacToeGame(player1, player2, iterations)
        game.play()