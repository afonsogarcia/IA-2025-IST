# nuruomino.py: Template para implementação do projeto de Inteligência Artificial 2024/2025.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 17:
# 96883 Jose Afonso Garcia
# 96914 Tomas Antunes

from search import Problem, Node, astar_search
from sys import stdin
from copy import deepcopy
from utils import *

class Region:
    """Classe para cachear informações sobre uma região específica."""
    
    def __init__(self, region_id, positions, size):
        self.region_id = region_id
        self.positions = set(positions)  
        self.size = size
        
        # Cache de possibilidades para cada estado do tabuleiro
        self._valid_placements_cache = {}  
        self._mrv_cache = {}  
        
    def get_board_state_hash(self, board):
        """Gera um hash do estado atual do tabuleiro para usar como chave de cache."""
        relevant_positions = []
        
        # Incluir posições preenchidas na região
        for row, col in self.positions:
            if board.is_position_filled(row, col):
                relevant_positions.append((row, col, board.get_filled_value(row, col)))
        
        # Incluir posições preenchidas adjacentes à região (para verificação de adjacência)
        for row, col in self.positions:
            for adj_row, adj_col in board.adjacent_positions(row, col):
                if (adj_row, adj_col) not in self.positions and board.is_position_filled(adj_row, adj_col):
                    relevant_positions.append((adj_row, adj_col, board.get_filled_value(adj_row, adj_col)))
        
        return hash(tuple(sorted(relevant_positions)))
    
    def get_valid_placements(self, board):
        """Retorna todas as colocações válidas para esta região, usando cache."""
        board_hash = self.get_board_state_hash(board)
        
        if board_hash not in self._valid_placements_cache:
            valid_placements = {}
            
            # Se a região já está preenchida, não há colocações válidas
            if self.region_id in board.placed_pieces:
                self._valid_placements_cache[board_hash] = {}
                return {}
            
            for tetromino_type, variants in ALL_TETROMINO_VARIANTS.items():
                valid_placements[tetromino_type] = []
                for variant in variants:
                    placement = board.can_place_shape_in_region(self.region_id, variant, tetromino_type)
                    if placement:
                        valid_placements[tetromino_type].append((variant, placement))
            
            self._valid_placements_cache[board_hash] = valid_placements
        
        return self._valid_placements_cache[board_hash]
    
    def get_mrv_count(self, board):
        """Retorna o número de opções válidas para esta região (para MRV)."""
        board_hash = self.get_board_state_hash(board)
        
        if board_hash not in self._mrv_cache:
            valid_placements = self.get_valid_placements(board)
            count = 0
            for tetromino_type, placements in valid_placements.items():
                if placements:  # Se há pelo menos uma colocação válida para este tipo
                    count += 1
            self._mrv_cache[board_hash] = count
        
        return self._mrv_cache[board_hash]
    

class NuruominoState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = NuruominoState.state_id
        NuruominoState.state_id += 1

    def __lt__(self, other):
        """Método utilizado para desempate na gestão da lista de abertos."""
        return self.id < other.id


class Board:
    def __init__(self, grid, filled_grid=None, placed_pieces=None):
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
        self._region_objects = {}  # region_id -> Region object
        
        # Inicializar cache de regiões
        self._build_regions_cache()

    def _build_regions_cache(self):
        """Constrói cache de regiões para otimização."""
        if self._regions_cache is None:
            # Usar multimap do utils para agrupar posições por região
            region_position_pairs = []
            for row in range(self.rows):
                for col in range(self.cols):
                    region = self.grid[row][col]
                    region_position_pairs.append((region, (row, col)))
            self._regions_cache = multimap(region_position_pairs)
            
            # Criar objetos Region para cada região
            for region_id, positions in self._regions_cache.items():
                self._region_objects[region_id] = Region(region_id, positions, len(positions))

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



    def adjacent_positions(self, row: int, col: int) -> list:
        """Devolve as posições adjacentes à célula, em todas as direções,
        incluindo diagonais."""
        adjacent = []
        
        
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        for dr, dc in directions:
            adj_row, adj_col = row + dr, col + dc
            
            # Verificar se está dentro dos limites do tabuleiro
            if 0 <= adj_row < self.rows and 0 <= adj_col < self.cols:
                adjacent.append((adj_row, adj_col))
        
        return adjacent

    def adjacent_values(self, row: int, col: int) -> list:
        """Devolve os valores das células adjacentes à célula,
        em todas as direções, incluindo diagonais."""
        values = []
        
        
        for adj_row, adj_col in self.adjacent_positions(row, col):
            
            if self.is_position_filled(adj_row, adj_col):
                values.append(self.filled_grid[adj_row][adj_col])
            else:
                
                values.append(self.grid[adj_row][adj_col])
        
        return values

    def adjacent_regions(self, region: int) -> list:
        """Devolve uma lista das regiões que fazem fronteira com a região."""
        if region not in self._adjacent_regions_cache:
            
            region_positions = self.get_region_positions(region)
            adjacent_values_set = set()
            
            
            for row, col in region_positions:
                
                for adj_row, adj_col in self.adjacent_positions(row, col):
                    
                    if self.grid[adj_row][adj_col] != region:
                        
                        adjacent_values_set.add(self.grid[adj_row][adj_col])
            
            # Converter para lista e guardar no cache
            self._adjacent_regions_cache[region] = list(adjacent_values_set)
        
        return self._adjacent_regions_cache[region]

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
        new_board = Board(
            self.grid, 
            deepcopy(self.filled_grid),        deepcopy(self.placed_pieces)
        )
        # Compartilhar os objetos Region (eles têm seu próprio cache)
        new_board._region_objects = self._region_objects
        return new_board

    def can_place_shape_in_region(self, region: int, shape: list, tetromino_type: str):
        """Validação aprimorada de posicionamento de formas - permite tetrominos em regiões maiores."""
        region_positions = set(self.get_region_positions(region))
        
        
        if len(region_positions) < 4:
            return None  
        
        
        for start_row, start_col in region_positions:
            shape_positions = []
            valid_placement = True
            
            for dx, dy in shape:
                new_row, new_col = start_row + dx, start_col + dy
                shape_positions.append((new_row, new_col))
                
                
                if ((new_row, new_col) not in region_positions or 
                    self.is_position_filled(new_row, new_col)):
                    valid_placement = False
                    break
            
            if (valid_placement and 
                len(set(shape_positions)) == len(shape) and
                len(shape_positions) == 4):  
                
                
                if (self._is_tetromino_connected(shape_positions) and
                    not self.would_create_2x2_block(shape_positions) and
                    not self.would_create_adjacent_same_pieces(shape_positions, tetromino_type)):
                    return shape_positions
        
        return None
    
    def _is_tetromino_connected(self, positions):
        if len(positions) != 4:
            return False
        
        positions_set = set(positions)
        start = positions[0]
        visited = {start}
        queue = [start]
        
        while queue:
            row, col = queue.pop(0)
            for dr, dc in orientations:
                adj_row, adj_col = row + dr, col + dc
                if ((adj_row, adj_col) in positions_set and 
                    (adj_row, adj_col) not in visited):
                    visited.add((adj_row, adj_col))
                    queue.append((adj_row, adj_col))
        
        return len(visited) == 4
    
    def would_create_2x2_block(self, new_positions: list) -> bool:
        """Verifica se novas posições criariam um bloco 2x2."""
        filled_positions = self.get_filled_positions()
        all_positions = filled_positions + new_positions
        positions_set = set(all_positions)
        
        for row, col in new_positions:
            patterns = [
                [(row, col), (row, col+1), (row+1, col), (row+1, col+1)],
                [(row-1, col), (row-1, col+1), (row, col), (row, col+1)],
                [(row-1, col-1), (row-1, col), (row, col-1), (row, col)],
                [(row, col-1), (row, col), (row+1, col-1), (row+1, col)]            ]
            
            for pattern in patterns:
                # Primeiro, verificar se TODAS as posições do padrão estão dentro dos limites
                if all(0 <= r < len(self.grid) and 0 <= c < len(self.grid[0]) for r, c in pattern):
                    # Depois, verificar se TODAS essas posições estão preenchidas
                    if all((r, c) in positions_set for r, c in pattern):
                        return True
                        
        return False

    def would_create_adjacent_same_pieces(self, new_positions: list, tetromino_type: str) -> bool:
        """Verifica se a colocação criaria peças iguais ortogonalmente adjacentes."""
        for row, col in new_positions:
            for dr, dc in orientations:
                adj_row, adj_col = row + dr, col + dc
                if (0 <= adj_row < self.rows and 0 <= adj_col < self.cols and
                    self.is_position_filled(adj_row, adj_col) and
                    self.get_filled_value(adj_row, adj_col) == tetromino_type):
                    return True
        return False

    def is_connected(self) -> bool:
        """Verifica se todas as peças preenchidas estão ortogonalmente conectadas."""
        filled_positions = set(self.get_filled_positions())
        
        if not filled_positions:
            return True
        # BFS para verificar conectividade
        start = first(filled_positions)
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
    
    def get_available_regions_mrv(self) -> list:
        """Retorna regiões disponíveis ordenadas por MRV (Most Restrictive Variable)."""
        available_regions = []
        
        for region in sorted(self.get_all_regions()):  # Ordenação estável por ID da região
            if region not in self.placed_pieces:
                # Usar o objeto Region para obter o número de opções
                region_obj = self._region_objects[region]
                num_options = region_obj.get_mrv_count(self)
                available_regions.append((region, num_options))
        
        # Ordenar por MRV (menor número de opções primeiro) e em caso de empate pelo ID da região
        available_regions.sort(key=lambda x: (x[1], x[0]))
        return available_regions

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
                row = [num_or_str(x) for x in line.split('\t')]
                grid.append(row)
        return Board(grid)


# Definições dos tetraminos
ALL_TETROMINO_VARIANTS = {
    'L': [
        [(0, 0), (-1, 0), (-1, 1), (-1, 2)],
        [(0, 0), (0, -1), (1, -1), (2, -1)],
        [(0, 0), (0, -1), (-1, -1), (-2, -1)],
        [(0, 0), (-1, 0), (-1, -1), (-1, -2)],
        [(0, 0), (0, 1), (-1, 1), (-2, 1)],
        [(0, 0), (0, 1), (1, 1), (2, 1)],
        [(0, 0), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (1, 0), (1, -1), (1, -2)],
    ],
    'I': [
        [(0, 0), (1, 0), (2, 0), (3, 0)],
        [(0, 0), (0, 1), (0, 2), (0, 3)],
    ],
    'T': [
        [(0, 0), (0, 1), (0, 2), (1, 1)],
        [(0, 0), (1, -1), (1, 0), (2, 0)],
        [(0, 0), (1, 0), (1, 1), (1, -1)],
        [(0, 0), (1, 0), (1, 1), (2, 0)],
    ],
    'S': [
        [(0, 0), (0, -1), (-1, -1), (-1, -2)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
        [(0, 0), (0, 1), (-1, 1), (-1, 2)],
        [(0, 0), (1, -1), (1, 0), (2, -1)],
    ]
}


class Nuruomino(Problem):
    """Classe principal do problema NURUOMINO - versão otimizada."""
    
    def __init__(self, initial_state: Board):
        """O construtor especifica o estado inicial."""
        state = NuruominoState(initial_state)
        super().__init__(state)
        
    def actions(self, state: NuruominoState):
        """Optimized action generation with MRV and simplified forward checking."""
        board = state.board
        best_region = None
        best_actions = None
        min_actions = float('inf')


        available_regions = board.get_available_regions_mrv()
        
        for region_id, mrv_count in available_regions:
            if mrv_count == 0:
                return [] 
            
            region_positions = set(board.get_region_positions(region_id))
            actions_for_region = []

            for piece_letter, shapes in ALL_TETROMINO_VARIANTS.items():
                for index, shape in enumerate(shapes):
                    for origin in region_positions:
                        shape_abs = [(origin[0] + dx, origin[1] + dy) for dx, dy in shape]
                        

                        if all(pos in region_positions and not board.is_position_filled(pos[0], pos[1]) 
                                for pos in shape_abs):
                            

                            if (not board.would_create_adjacent_same_pieces(shape_abs, piece_letter) and 
                                not board.would_create_2x2_block(shape_abs)):
                                
                                
                                valid_action = True 
                                
                                if valid_action:
                                    actions_for_region.append((region_id, piece_letter, shape, index, shape_abs))
            
            
            if 0 < len(actions_for_region) <= 1:
                return actions_for_region
            
            
            if 0 < len(actions_for_region) < min_actions:
                min_actions = len(actions_for_region)
                best_actions = actions_for_region
                
                
                if min_actions <= 3:
                    break 
        
        return best_actions if best_actions else []
    
    def result(self, state: NuruominoState, action):
        """Retorna o estado resultante de executar a 'action'."""
        region_id, piece_letter, shape, index, shape_abs = action

        
        new_board = state.board.copy()
        
        
        for i, j in shape_abs:
            new_board.filled_grid[i][j] = piece_letter
        
        
        new_board.placed_pieces[region_id] = piece_letter
        
        return NuruominoState(new_board)

    def goal_test(self, state: NuruominoState) -> bool:
        """Retorna True se o estado é um estado objetivo."""
        board = state.board
        
        
        if len(board.placed_pieces) != len(board.get_all_regions()):
            return False
            
        
        if not board.is_connected():
            return False
            
        
        return True
        
    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        board = node.state.board
        unplaced = len(board.get_all_regions()) - len(board.placed_pieces)
        available = board.get_available_regions_mrv()
        counts = [opt for _, opt in available]
        h_val = unplaced
        if counts:
            most_freq = mode(counts)     
            h_val += most_freq * 0.01     
        return h_val



def solve_nuruomino():
    """resolve o problema nuruomino"""
    board = Board.parse_instance()
    problem = Nuruomino(board)
    
    solution = astar_search(problem)
    
    if solution:
        solution.state.board.print_board()
    else:
        print("Nenhuma Solução Encontrada")


if __name__ == "__main__":
    solve_nuruomino()