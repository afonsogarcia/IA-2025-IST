# nuruomino.py: Template para implementação do projeto de Inteligência Artificial 2024/2025.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 17:
# 96883 Jose Afonso Garcia
# 96914 Tomas Antunes

from search import Problem, Node, astar_search, greedy_search
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
            # Usar multimap do utils para agrupar posições por região
            region_position_pairs = []
            for row in range(self.rows):
                for col in range(self.cols):
                    region = self.grid[row][col]
                    region_position_pairs.append((region, (row, col)))
            self._regions_cache = multimap(region_position_pairs)

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

    def adjacent_positions(self, row: int, col: int) -> list:
        """Devolve as posições adjacentes à célula, em todas as direções,
        incluindo diagonais."""
        adjacent = []
        
        # Direções: ortogonais e diagonais (8 direções)
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
        
        # Obter posições adjacentes
        for adj_row, adj_col in self.adjacent_positions(row, col):
            # Se a posição estiver preenchida, usar o valor preenchido
            if self.is_position_filled(adj_row, adj_col):
                values.append(self.filled_grid[adj_row][adj_col])
            else:
                # Caso contrário, usar o número da região
                values.append(self.grid[adj_row][adj_col])
        
        return values

    def adjacent_regions(self, region: int) -> list:
        """Devolve uma lista das regiões que fazem fronteira com a região."""
        if region not in self._adjacent_regions_cache:
            # Obter todas as posições da região
            region_positions = self.get_region_positions(region)
            adjacent_values_set = set()
            
            # Para cada posição na região
            for row, col in region_positions:
                # Obter valores adjacentes (inclui diagonais)
                for adj_row, adj_col in self.adjacent_positions(row, col):
                    # Verificar se é outra região (não é da região atual)
                    if self.grid[adj_row][adj_col] != region:
                        # Adicionar a região adjacente ao conjunto
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
        return Board(
            self.grid,  # Grid original não muda
            deepcopy(self.filled_grid),
            deepcopy(self.placed_pieces)
        )
        
    def can_place_shape_in_region(self, region: int, shape: list, tetromino_type: str):
        """Verifica se uma forma pode ser colocada numa região específica."""
        region_positions = set(self.get_region_positions(region))
        
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
                if self.is_position_filled(new_row, new_col):
                    valid_placement = False
                    break
            
            if valid_placement and len(set(shape_positions)) == len(shape):
                # Verificações de validade
                if (not self.would_create_2x2_block(shape_positions) and
                    not self.would_create_adjacent_same_pieces(shape_positions, tetromino_type)):
                    return shape_positions
        
        return None
    
    def would_create_2x2_block(self, new_positions: list) -> bool:
        """Verifica se colocar as peças criaria um bloco 2x2."""
        temp_filled = set(self.get_filled_positions())
        temp_filled.update(new_positions)
        
        # Para cada nova posição, verificar apenas os possíveis blocos 2x2 que ela poderia formar
        for row, col in new_positions:
            # Uma posição só pode formar um bloco 2x2 se for um dos cantos do bloco
            
            # Verificar se esta posição é o canto superior esquerdo de um bloco 2x2
            if ((row+1, col) in temp_filled and (row, col+1) in temp_filled and 
                (row+1, col+1) in temp_filled):
                return True
                
            # Verificar se esta posição é o canto superior direito de um bloco 2x2
            if ((row+1, col) in temp_filled and (row, col-1) in temp_filled and 
                (row+1, col-1) in temp_filled):
                return True
                
            # Verificar se esta posição é o canto inferior esquerdo de um bloco 2x2
            if ((row-1, col) in temp_filled and (row, col+1) in temp_filled and 
                (row-1, col+1) in temp_filled):
                return True
                
            # Verificar se esta posição é o canto inferior direito de um bloco 2x2
            if ((row-1, col) in temp_filled and (row, col-1) in temp_filled and 
                (row-1, col-1) in temp_filled):
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
                # Contar quantas formas podem ser colocadas nesta região
                possible_placements = 0
                
                for tetromino_type, variants in ALL_TETROMINO_VARIANTS.items():
                    for variant in variants:
                        if self.can_place_shape_in_region(region, variant, tetromino_type):
                            possible_placements += 1
                            break  # Uma variante é suficiente para este tipo
                
                available_regions.append((region, possible_placements))
        
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



def generate_optimized_variants(tetromino_name, base_shape):
    """Gera variantes otimizadas por tipo de tetromino incluindo reflexões."""
    variants = []
    
    if tetromino_name == 'I':
        # I: apenas 2 orientações únicas (horizontal e vertical)
        # Reflexões são idênticas às rotações
        variants.append(base_shape)  # horizontal
        variants.append(rotate_90(base_shape))  # vertical
        
    elif tetromino_name == 'S':
        # S: 4 orientações únicas (incluindo reflexões que formam Z)
        current = base_shape
        # 2 rotações normais
        for _ in range(2):
            variants.append(current)
            current = rotate_90(current)
        
        # 2 rotações refletidas (formam Z)
        reflected = reflect_horizontal(base_shape)
        current = reflected
        for _ in range(2):
            variants.append(current)
            current = rotate_90(current)
        
    elif tetromino_name == 'T':
        # T: 4 orientações únicas (reflexões são idênticas às rotações)
        current = base_shape
        for _ in range(4):
            variants.append(current)
            current = rotate_90(current)
            
    elif tetromino_name == 'L':
        # L: 8 orientações únicas (4 rotações + 4 reflexões)
        current = base_shape
        # 4 rotações normais
        for _ in range(4):
            variants.append(current)
            current = rotate_90(current)
        
        # 4 rotações refletidas
        reflected = reflect_horizontal(base_shape)
        current = reflected
        for _ in range(4):
            variants.append(current)
            current = rotate_90(current)
    
    # Remover duplicatas usando set e tuple
    unique_variants = []
    seen = set()
    for variant in variants:
        variant_tuple = tuple(sorted(variant))
        if variant_tuple not in seen:
            seen.add(variant_tuple)
            unique_variants.append(variant)
    
    return unique_variants

# Cache de variantes geradas uma única vez usando versão otimizada
ALL_TETROMINO_VARIANTS = {}
for name, shape in TETROMINOS.items():
    ALL_TETROMINO_VARIANTS[name] = generate_optimized_variants(name, shape)


class Nuruomino(Problem):
    """Classe principal do problema NURUOMINO - versão otimizada."""
    
    def __init__(self, initial_state: Board):
        """O construtor especifica o estado inicial."""
        state = NuruominoState(initial_state)
        super().__init__(state)
        
    def actions(self, state: NuruominoState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        actions = []
        board = state.board
        available_regions = board.get_available_regions_mrv()
        
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
                placement = board.can_place_shape_in_region(target_region, variant, tetromino_type)
                if placement:
                    action = (target_region, tetromino_type, placement)
                    actions.append(action)
                    
        return actions

    def result(self, state: NuruominoState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
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
        if not board.is_connected():
            return False
            
        # As outras verificações (2x2 e adjacência) já são feitas durante a colocação
        return True
        
    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        state = node.state
        board = state.board
        
        # Número de regiões não preenchidas (admissível)
        unplaced_regions = len(board.get_all_regions()) - len(board.placed_pieces)
        
        # Penalização por regiões com poucas opções (mantém admissibilidade)
        available_regions = board.get_available_regions_mrv()
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
    
    # Resolver usando astar_search
    solution = greedy_search(problem)
    
    if solution:
        solution.state.board.print_board()
    else:
        print("Nenhuma solução encontrada")


if __name__ == "__main__":
    solve_nuruomino()