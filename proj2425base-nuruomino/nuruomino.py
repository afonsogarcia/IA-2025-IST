# nuruomino_optimized.py: Versão otimizada baseada nas boas práticas do pipe.py
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.

# Grupo 00:
# 00000 Nome1
# 00000 Nome2

from search import Problem, Node, greedy_search
from sys import stdin
from copy import deepcopy
from utils import *

class NuruominoState:
    """Estado do puzzle NURUOMINO - versão otimizada."""
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = NuruominoState.state_id
        NuruominoState.state_id += 1

    def __lt__(self, other):
        """Método utilizado para desempate na gestão da lista de abertos."""
        return self.id < other.id


class Board:
    """Representação interna otimizada de um tabuleiro do Puzzle Nuruomino."""
    
    def __init__(self, grid, filled_grid=None, placed_pieces=None):
        """Inicializa o tabuleiro com a grelha de regiões."""
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if grid else 0
        
        # Estado do jogo
        self.filled_grid = filled_grid or [[None for _ in range(self.cols)] for _ in range(self.rows)]
        self.placed_pieces = placed_pieces or {}  # region_id -> piece_type
        
        # Cache para otimização
        self._regions_cache = None
        self._region_positions_cache = {}
        self._adjacent_regions_cache = {}
        
        # Inicializar cache de regiões
        self._build_regions_cache()

    def _build_regions_cache(self):
        """Constrói cache de regiões para otimização."""
        if self._regions_cache is None:
            self._regions_cache = {}
            for row in range(self.rows):
                for col in range(self.cols):
                    region = self.grid[row][col]
                    if region not in self._regions_cache:
                        self._regions_cache[region] = []
                    self._regions_cache[region].append((row, col))

    def get_value(self, row: int, col: int):
        """Retorna o valor numa determinada posição."""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.grid[row][col]
        return None

    def get_filled_value(self, row: int, col: int):
        """Retorna o valor preenchido numa posição (peça colocada)."""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.filled_grid[row][col]
        return None

    def is_position_filled(self, row: int, col: int) -> bool:
        """Verifica se uma posição está preenchida."""
        return self.get_filled_value(row, col) is not None

    def get_region_positions(self, region: int) -> list:
        """Retorna todas as posições que pertencem a uma região."""
        if region not in self._region_positions_cache:
            self._region_positions_cache[region] = self._regions_cache.get(region, [])
        return self._region_positions_cache[region]

    def get_all_regions(self) -> list:
        """Retorna uma lista com todos os identificadores de regiões."""
        return list(self._regions_cache.keys())

    def get_region_size(self, region: int) -> int:
        """Retorna o número de células numa região."""
        return len(self.get_region_positions(region))

    def adjacent_regions(self, region: int) -> list:
        """Devolve uma lista das regiões que fazem fronteira com a região."""
        if region not in self._adjacent_regions_cache:
            adjacent = set()
            for row, col in self.get_region_positions(region):
                for dr, dc in orientations:
                    new_row, new_col = row + dr, col + dc
                    if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
                        neighbor_region = self.grid[new_row][new_col]
                        if neighbor_region != region:
                            adjacent.add(neighbor_region)
            self._adjacent_regions_cache[region] = list(adjacent)
        return self._adjacent_regions_cache[region]

    def get_orthogonal_neighbors(self, row: int, col: int) -> list:
        """Retorna as posições ortogonalmente adjacentes."""
        neighbors = []
        for dr, dc in orientations:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
                neighbors.append((new_row, new_col))
        return neighbors

    def get_filled_positions(self) -> list:
        """Retorna todas as posições preenchidas."""
        filled = []
        for row in range(self.rows):
            for col in range(self.cols):
                if self.is_position_filled(row, col):
                    filled.append((row, col))
        return filled   
        
    def copy(self):
        """Cria uma cópia profunda do tabuleiro."""
        return Board(
            self.grid,  # Grid original não muda
            deepcopy(self.filled_grid),
            deepcopy(self.placed_pieces)
        )

    def print_board(self):
        """Imprime o tabuleiro no formato de saída."""
        for row in range(self.rows):
            row_str = []
            for col in range(self.cols):
                if self.is_position_filled(row, col):
                    row_str.append(str(self.filled_grid[row][col]))
                else:
                    row_str.append(str(self.grid[row][col]))
            print('\t'.join(row_str))

    @staticmethod
    def parse_instance():
        """Lê a instância do problema do standard input."""
        grid = []
        for line in stdin:
            line = line.strip()
            if line:
                row = [int(x) for x in line.split('\t')]
                grid.append(row)
        return Board(grid)


# Definições dos tetraminós otimizadas
TETROMINOS = {
    'L': [(0, 0), (1, 0), (2, 0), (2, 1)],
    'I': [(0, 0), (1, 0), (2, 0), (3, 0)],
    'T': [(0, 0), (0, 1), (0, 2), (1, 1)],
    'S': [(0, 0), (0, 1), (1, 1), (1, 2)]
}

def rotate_90(shape):
    """Roda uma forma 90 graus no sentido horário."""
    return [(y, -x) for x, y in shape]

def reflect_horizontal(shape):
    """Reflete uma forma horizontalmente."""
    return [(-x, y) for x, y in shape]

def normalize_shape(shape):
    """Normaliza uma forma para que comece em (0, 0)."""
    if not shape:
        return shape
    min_x = min(x for x, y in shape)
    min_y = min(y for x, y in shape)
    return [(x - min_x, y - min_y) for x, y in shape]

def generate_all_variants(base_shape):
    """Gera todas as variantes (rotações + reflexões) de uma forma."""
    variants = set()
    current = base_shape
    
    # 4 rotações
    for _ in range(4):
        variants.add(tuple(sorted(normalize_shape(current))))
        # Adicionar reflexão de cada rotação
        reflected = reflect_horizontal(current)
        variants.add(tuple(sorted(normalize_shape(reflected))))
        current = rotate_90(current)
    
    return [list(variant) for variant in variants]

# Cache de variantes geradas uma única vez
ALL_TETROMINO_VARIANTS = {}
for name, shape in TETROMINOS.items():
    ALL_TETROMINO_VARIANTS[name] = generate_all_variants(shape)


class Nuruomino(Problem):
    """Classe principal do problema NURUOMINO - versão otimizada."""
    
    def __init__(self, board: Board):
        """Construtor que especifica o estado inicial."""
        initial_state = NuruominoState(board)
        super().__init__(initial_state)

    def _can_place_shape_in_region(self, board: Board, region: int, shape: list, tetromino_type: str):
        """Verifica se uma forma pode ser colocada numa região específica."""
        region_positions = set(board.get_region_positions(region))
        
        if len(shape) > len(region_positions):
            return None
        
        # Tentar colocar a forma em cada posição possível da região
        for start_row, start_col in region_positions:
            shape_positions = []
            valid_placement = True
            
            for dx, dy in shape:
                new_row, new_col = start_row + dx, start_col + dy
                shape_positions.append((new_row, new_col))
                
                # Verificar se a posição está dentro da região
                if (new_row, new_col) not in region_positions:
                    valid_placement = False
                    break
                
                # Verificar se a posição já está ocupada
                if board.is_position_filled(new_row, new_col):
                    valid_placement = False
                    break
            
            if valid_placement and len(set(shape_positions)) == len(shape):
                # Verificações de validade
                if (not self._would_create_2x2_block(board, shape_positions) and
                    not self._would_create_adjacent_same_pieces(board, shape_positions, tetromino_type)):
                    return shape_positions
        
        return None

    def _would_create_2x2_block(self, board: Board, new_positions: list) -> bool:
        """Verifica se colocar as peças criaria um bloco 2x2."""
        temp_filled = set(board.get_filled_positions())
        temp_filled.update(new_positions)
        
        for row in range(board.rows - 1):
            for col in range(board.cols - 1):
                block_positions = [
                    (row, col), (row, col + 1),
                    (row + 1, col), (row + 1, col + 1)
                ]
                if all(pos in temp_filled for pos in block_positions):
                    return True
        
        return False

    def _would_create_adjacent_same_pieces(self, board: Board, new_positions: list, tetromino_type: str) -> bool:
        """Verifica se a colocação criaria peças iguais ortogonalmente adjacentes."""
        for row, col in new_positions:
            for dr, dc in orientations:
                adj_row, adj_col = row + dr, col + dc
                if (0 <= adj_row < board.rows and 0 <= adj_col < board.cols and
                    board.is_position_filled(adj_row, adj_col) and
                    board.get_filled_value(adj_row, adj_col) == tetromino_type):
                    return True
        return False

    def _is_connected(self, board: Board) -> bool:
        """Verifica se todas as peças preenchidas estão ortogonalmente conectadas."""
        filled_positions = set(board.get_filled_positions())
        
        if not filled_positions:
            return True
        
        # BFS para verificar conectividade
        start = next(iter(filled_positions))
        visited = {start}
        queue = [start]
        
        while queue:
            row, col = queue.pop(0)
            for dr, dc in orientations:
                adj_row, adj_col = row + dr, col + dc
                if ((adj_row, adj_col) in filled_positions and 
                    (adj_row, adj_col) not in visited):
                    visited.add((adj_row, adj_col))
                    queue.append((adj_row, adj_col))
        
        return len(visited) == len(filled_positions)

    def _get_available_regions_mrv(self, board: Board) -> list:
        """Retorna regiões disponíveis ordenadas por MRV (Most Restrictive Variable)."""
        available_regions = []
        
        for region in board.get_all_regions():
            if region not in board.placed_pieces:
                # Contar quantas formas podem ser colocadas nesta região
                possible_placements = 0
                
                for tetromino_type, variants in ALL_TETROMINO_VARIANTS.items():
                    for variant in variants:
                        if self._can_place_shape_in_region(board, region, variant, tetromino_type):
                            possible_placements += 1
                            break  # Uma variante é suficiente para este tipo
                
                available_regions.append((region, possible_placements))
        
        # Ordenar por MRV (menor número de opções primeiro)
        available_regions.sort(key=lambda x: x[1])
        return available_regions

    def actions(self, state: NuruominoState):
        """Retorna uma lista de ações que podem ser executadas."""
        actions = []
        board = state.board
        available_regions = self._get_available_regions_mrv(board)
        
        if not available_regions:
            return actions
        
        # MRV: focar na região com menos opções
        target_region, num_options = available_regions[0]
        
        # Se nenhuma opção disponível, retornar lista vazia (estado inválido)
        if num_options == 0:
            return actions
        
        # Tentar todas as peças e variantes para esta região
        for tetromino_type, variants in ALL_TETROMINO_VARIANTS.items():
            for variant in variants:
                placement = self._can_place_shape_in_region(board, target_region, variant, tetromino_type)
                if placement:
                    action = (target_region, tetromino_type, placement)
                    actions.append(action)
        
        return actions

    def result(self, state: NuruominoState, action):
        """Retorna o estado resultante de executar a ação."""
        region, tetromino_type, positions = action
        
        # Criar novo tabuleiro
        new_board = state.board.copy()
        
        # Colocar a peça nas posições especificadas
        for row, col in positions:
            new_board.filled_grid[row][col] = tetromino_type
        
        # Marcar a região como preenchida
        new_board.placed_pieces[region] = tetromino_type
        
        return NuruominoState(new_board)

    def goal_test(self, state: NuruominoState) -> bool:
        """Retorna True se o estado é um estado objetivo."""
        board = state.board
        
        # Verificar se todas as regiões estão preenchidas
        if len(board.placed_pieces) != len(board.get_all_regions()):
            return False
        
        # Verificar conectividade
        if not self._is_connected(board):
            return False
        
        # As outras verificações (2x2 e adjacência) já são feitas durante a colocação
        return True

    def h(self, node: Node):
        """Função heurística admissível para A*."""
        state = node.state
        board = state.board
        
        # Número de regiões não preenchidas (admissível)
        unplaced_regions = len(board.get_all_regions()) - len(board.placed_pieces)
        
        # Penalização por regiões com poucas opções (mantém admissibilidade)
        available_regions = self._get_available_regions_mrv(board)
        constraint_penalty = 0
        
        for region, num_options in available_regions:
            if num_options == 0:
                return float('inf')  # Estado inválido
            elif num_options == 1:
                constraint_penalty += 0.1  # Pequena penalização para regiões muito restritivas
        
        return unplaced_regions + constraint_penalty


# Função principal otimizada baseada no padrão do pipe.py
def solve_nuruomino():
    """Resolve o puzzle NURUOMINO."""
    # Ler tabuleiro
    board = Board.parse_instance()
    
    # Criar problema
    problem = Nuruomino(board)
    
    # Resolver usando greedy_search (como no pipe.py)
    solution = greedy_search(problem)
    
    if solution:
        solution.state.board.print_board()
    else:
        print("Nenhuma solução encontrada")


if __name__ == "__main__":
    solve_nuruomino()