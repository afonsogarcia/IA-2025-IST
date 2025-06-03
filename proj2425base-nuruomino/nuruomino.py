# nuruomino.py: Template para implementação do projeto de Inteligência Artificial 2024/2025.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 00:
# 00000 Nome1
# 00000 Nome2

from search import Problem, Node
from sys import stdin
from utils import *  # Importamos todas as funções disponíveis em utils.py

class NuruominoState:
    state_id = 0

    def __init__(self, board, filled_grid=None, placed_pieces=None):
        self.board = board  # Board original com regiões
        self.id = NuruominoState.state_id
        NuruominoState.state_id += 1
        
        # Grid com as peças colocadas (None = vazio, 'L'/'I'/'T'/'S' = peça)
        if filled_grid is None:
            self.filled_grid = [[None for _ in range(board.cols)] for _ in range(board.rows)]
        else:
            self.filled_grid = [row[:] for row in filled_grid]  # Deep copy
        
        # Dicionário das peças colocadas por região {region_id: tetromino_type}
        self.placed_pieces = placed_pieces.copy() if placed_pieces else {}

    def __lt__(self, other):
        """ Este método é utilizado em caso de empate na gestão da lista
        de abertos nas procuras informadas. """
        return self.id < other.id
    
    def clone(self):
        """Cria uma cópia profunda do estado."""
        return NuruominoState(self.board, self.filled_grid, self.placed_pieces)

class Board:
    """Representação interna de um tabuleiro do Puzzle Nuruomino."""
    
    def __init__(self, grid):
        """Inicializa o tabuleiro com a grelha de regiões."""
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if grid else 0
        # Criar um mapeamento de regiões para posições
        self.regions = {}
        for row in range(self.rows):
            for col in range(self.cols):
                region = self.grid[row][col]
                if region not in self.regions:
                    self.regions[region] = []
                self.regions[region].append((row, col))
    
    def is_position_filled(self, state, row, col):
        """Verifica se uma posição está preenchida."""
        return state.filled_grid[row][col] is not None
    
    def get_filled_positions(self, state):
        """Retorna todas as posições preenchidas."""
        filled = []
        for row in range(self.rows):
            for col in range(self.cols):
                if self.is_position_filled(state, row, col):
                    filled.append((row, col))
        return filled

    def get_state_value(self, state, row: int, col: int):
        """Retorna o valor numa determinada posição, considerando peças colocadas."""
        # Ajustar coordenadas para 0-based se necessário
        adj_row, adj_col = row - 1, col - 1
        
        if 0 <= adj_row < self.rows and 0 <= adj_col < self.cols:
            # Se há uma peça colocada nesta posição, retornar o tipo da peça
            if self.is_position_filled(state, adj_row, adj_col):
                return state.filled_grid[adj_row][adj_col]
            else:
                # Se não há peça, retornar o valor original da região
                return self.grid[adj_row][adj_col]
        return None
    
    def get_state_adjacent_values(self, state, row: int, col: int) -> list:
        """Retorna os valores das células adjacentes à posição, considerando peças colocadas."""
        values = []
        
        # Converter coordenadas baseadas em 1 para baseadas em 0
        adj_row, adj_col = row - 1, col - 1
        
        for pos_row, pos_col in self.adjacent_positions(adj_row, adj_col):
            # Se a posição tem uma peça colocada, retornar o tipo da peça
            if self.is_position_filled(state, pos_row, pos_col):
                values.append(state.filled_grid[pos_row][pos_col])
            else:
                # Se não tem peça, retornar o valor original da região
                values.append(self.grid[pos_row][pos_col])
        
        return values

    def print_state_board(self, state):
        """Imprime o tabuleiro atual mostrando as peças colocadas."""
        print("Board atual com peças colocadas:")
        for row in range(self.rows):
            for col in range(self.cols):
                if self.is_position_filled(state, row, col):
                    # Mostrar o tipo da peça (L, I, T, S)
                    print(state.filled_grid[row][col], end='\t')
                else:
                    # Mostrar o número da região original
                    print(self.grid[row][col], end='\t')
            print()  # Nova linha no final de cada row
        print()  # Linha em branco no final
    
    def get_value(self, row: int, col: int):
        """Retorna o valor numa determinada posição."""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.grid[row-1][col-1]
        return None
    
    def adjacent_regions(self, region: int) -> list:
        """Devolve uma lista das regiões que fazem fronteira com a região enviada no argumento."""
        adjacent = set()
        
        # Para cada posição da região
        for row, col in self.regions.get(region, []):
            # Verificar vizinhos ortogonais usando orientations do utils.py
            for dr, dc in orientations:
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
                    neighbor_region = self.grid[new_row][new_col]
                    if neighbor_region != region:
                        adjacent.add(neighbor_region)
        
        return list(adjacent)
    
    def adjacent_positions(self, row: int, col: int) -> list:
        """Devolve as posições adjacentes à posição, em todas as direções, incluindo diagonais."""
        positions = []
        
        # Verificar todas as 8 direções (incluindo diagonais)
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:  # Pular a própria posição
                    continue
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
                    positions.append((new_row, new_col))
        
        return positions
        
    def adjacent_values(self, row: int, col: int) -> list:
        """Devolve os valores das células adjacentes à posição, em todas as direções, incluindo diagonais."""
        values = []
        
        for pos_row, pos_col in self.adjacent_positions(row, col):
            values.append(self.grid[pos_row][pos_col])
        
        return values
    
    def get_region_positions(self, region: int) -> list:
        """Retorna todas as posições que pertencem a uma região."""
        return self.regions.get(region, [])
    
    def get_all_regions(self) -> list:
        """Retorna uma lista com todos os identificadores de regiões."""
        return list(self.regions.keys())
        
    def print_board(self):
        """Imprime o tabuleiro no formato original."""
        # Usando a função print_table do utils.py
        print_table(self.grid, sep='\t')
    
    @staticmethod
    def parse_instance():
        """Lê a instância do problema do standard input (stdin)
        e retorna uma instância da classe Board.

        Por exemplo:
            $ python3 pipe.py < test-01.txt

            > from sys import stdin
            > line = stdin.readline().split()
        """
        grid = []
        
        # Ler todas as linhas do stdin
        try:
            while True:
                line = stdin.readline()
                if not line:  # EOF
                    break
                line = line.strip()
                if line:  # Ignorar linhas vazias
                    # Dividir a linha por tabs e converter para inteiros
                    row = [int(x) for x in line.split('\t')]
                    grid.append(row)
        except EOFError:
            pass
        
        return Board(grid)

    @staticmethod
    def parse_from_file(filename):
        """Lê a instância do problema de um arquivo
        e retorna uma instância da classe Board.
        """
        grid = []
        
        try:
            with open(filename, 'r') as file:
                for line in file:
                    line = line.strip()
                    if line:  # Ignorar linhas vazias
                        # Dividir a linha por tabs e converter para inteiros
                        row = [int(x) for x in line.split('\t')]
                        grid.append(row)
        except FileNotFoundError:
            print(f"Erro: Arquivo {filename} não encontrado.")
            return None
        except Exception as e:
            print(f"Erro ao ler arquivo {filename}: {e}")
            return None
        
        return Board(grid)

    def is_valid_position(self, row: int, col: int) -> bool:
        """Verifica se uma posição é válida no tabuleiro."""
        return 0 <= row < self.rows and 0 <= col < self.cols
    
    def get_region_size(self, region: int) -> int:
        """Retorna o número de células numa região."""
        return len(self.regions.get(region, []))
    
    def get_region_for_position(self, row: int, col: int) -> int:
        """Retorna o identificador da região que contém a posição (row, col)."""
        if self.is_valid_position(row, col):
            return self.grid[row][col]
        return None
        
    def get_orthogonal_neighbors(self, row: int, col: int) -> list:
        """Retorna as posições ortogonalmente adjacentes (sem diagonais)."""
        neighbors = []
        # Usando orientations do utils.py (EAST, NORTH, WEST, SOUTH)
        for dr, dc in orientations:
            new_row, new_col = row + dr, col + dc
            if self.is_valid_position(new_row, new_col):
                neighbors.append((new_row, new_col))
        return neighbors
    
    def get_orthogonal_values(self, row: int, col: int) -> list:
        """Retorna os valores das células ortogonalmente adjacentes."""
        values = []
        for pos_row, pos_col in self.get_orthogonal_neighbors(row, col):
            values.append(self.grid[pos_row][pos_col])
        return values

# Definições dos tetraminós (formas básicas)
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
    # Usando a função utilitária min de forma mais eficiente
    xs = [x for x, y in shape]
    ys = [y for x, y in shape]
    min_x = min(xs)
    min_y = min(ys)
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

# Gerar todas as variantes de cada tetraminó
ALL_TETROMINO_VARIANTS = {}
for name, shape in TETROMINOS.items():
    ALL_TETROMINO_VARIANTS[name] = generate_all_variants(shape)

class Nuruomino(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        initial_state = NuruominoState(board)
        super().__init__(initial_state)
        self.board = board

    def _can_place_shape_in_region(self, state, region, shape, tetromino_type):
        """Verifica se uma forma pode ser colocada numa região específica."""
        region_positions = set(self.board.get_region_positions(region))
        
        # O tetromino deve caber dentro da região, mas não precisa preenchê-la completamente
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
                if self.board.is_position_filled(state, new_row, new_col):
                    valid_placement = False
                    break
            
            if valid_placement and len(set(shape_positions)) == len(shape):
                # Verificar se não cria blocos 2x2
                if not self._would_create_2x2_block(state, shape_positions, tetromino_type):
                    # Verificar se não há peças iguais adjacentes
                    if not self._would_create_adjacent_same_pieces(state, shape_positions, tetromino_type):
                        return shape_positions
        
        return None
    
    def _would_create_2x2_block(self, state, new_positions, tetromino_type):
        """Verifica se colocar as peças nas posições criaria um bloco 2x2."""
        # Criar um estado temporário com as novas posições
        temp_filled = set(self.board.get_filled_positions(state))
        temp_filled.update(new_positions)
        
        # Verificar todos os possíveis blocos 2x2
        for row in range(self.board.rows - 1):
            for col in range(self.board.cols - 1):
                block_positions = [
                    (row, col), (row, col + 1),
                    (row + 1, col), (row + 1, col + 1)
                ]
                
                if all(pos in temp_filled for pos in block_positions):
                    return True
        
        return False

    def _would_create_adjacent_same_pieces(self, state, new_positions, tetromino_type):
        """Verifica se a colocação criaria peças iguais ortogonalmente adjacentes."""
        for row, col in new_positions:
            # Verificar vizinhos ortogonais usando orientations do utils
            for dr, dc in orientations:
                adj_row, adj_col = row + dr, col + dc
                if (self.board.is_valid_position(adj_row, adj_col) and 
                   self.board.is_position_filled(state, adj_row, adj_col)):
                    # Verificar se é uma peça do mesmo tipo
                    if state.filled_grid[adj_row][adj_col] == tetromino_type:
                        return True
        
        return False
        
    def _is_connected(self, state):
        """Verifica se todas as peças preenchidas estão ortogonalmente conectadas."""
        filled_positions = set(self.board.get_filled_positions(state))
        
        if not filled_positions:
            return True
        
        # BFS para verificar conectividade usando funções do utils
        # Utilizamos first() para pegar o primeiro elemento do conjunto
        start = first(filled_positions)
        visited = {start}
        queue = [start]
        
        while queue:
            row, col = queue.pop(0)
            # Usando orientations (EAST, NORTH, WEST, SOUTH) do utils.py
            for dr, dc in orientations:
                adj_row, adj_col = row + dr, col + dc
                
                if ((adj_row, adj_col) in filled_positions and 
                    (adj_row, adj_col) not in visited):
                    visited.add((adj_row, adj_col))
                    queue.append((adj_row, adj_col))
        
        return len(visited) == len(filled_positions)
    
    def _get_available_regions(self, state):
        """Retorna regiões ainda não preenchidas, ordenadas por MRV."""
        available_regions = []
        
        for region in self.board.get_all_regions():
            if region not in state.placed_pieces:
                # Contar quantas formas podem ser colocadas nesta região
                possible_placements = 0
                
                for tetromino_type, variants in ALL_TETROMINO_VARIANTS.items():
                    # Verificar se pelo menos uma variante deste tipo pode ser colocada
                    can_place_type = False
                    for variant in variants:
                        if self._can_place_shape_in_region(state, region, variant, tetromino_type):
                            can_place_type = True
                            break  # Uma variante é suficiente para este tipo
                    
                    if can_place_type:
                        possible_placements += 1
                
                available_regions.append((region, possible_placements))
        
        # Ordenar por MRV (menor número de opções primeiro)
        available_regions.sort(key=lambda x: x[1])
        return [region for region, _ in available_regions]

    def actions(self, state):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        actions = []
        available_regions = self._get_available_regions(state)
        
        # Se não há regiões disponíveis, não há ações
        if not available_regions:
            return actions
        
        # Aplicar MRV: focar na região com menos opções
        target_region = available_regions[0]
        
        # Tentar todas as peças e variantes possíveis para esta região
        for tetromino_type, variants in ALL_TETROMINO_VARIANTS.items():
            for variant in variants:
                placement = self._can_place_shape_in_region(state, target_region, variant, tetromino_type)
                if placement:
                    action = (target_region, tetromino_type, variant, placement)
                    actions.append(action)
        
        return actions
    
    def result(self, state: NuruominoState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento."""
        
        # Suportar ambos os formatos de ação:
        # Formato do algoritmo: (region, tetromino_type, variant, positions)
        # Formato do exemplo: (region, tetromino_type, shape)
        if len(action) == 4:
            # Formato do algoritmo
            region, tetromino_type, variant, positions = action
        elif len(action) == 3:
            # Formato do exemplo da documentação
            region, tetromino_type, shape = action
            # Encontrar onde colocar a peça na região
            positions = self._can_place_shape_in_region(state, region, shape, tetromino_type)
            if positions is None:
                raise ValueError(f"Não é possível colocar a peça {tetromino_type} com forma {shape} na região {region}")
        else:
            raise ValueError(f"Formato de ação inválido: {action}")
        
        # Clonar o estado atual
        new_state = state.clone()
        
        # Colocar a peça nas posições especificadas
        for row, col in positions:
            new_state.filled_grid[row][col] = tetromino_type
        
        # Marcar a região como preenchida
        new_state.placed_pieces[region] = tetromino_type
        
        return new_state
        
    def goal_test(self, state: NuruominoState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo."""
        # Verificar se todas as regiões estão preenchidas
        if len(state.placed_pieces) != len(self.board.get_all_regions()):
            return False
        
        # Verificar se não há blocos 2x2
        filled_positions = set(self.board.get_filled_positions(state))
        for row in range(self.board.rows - 1):
            for col in range(self.board.cols - 1):
                block_positions = [
                    (row, col), (row, col + 1),
                    (row + 1, col), (row + 1, col + 1)
                ]
                if all(pos in filled_positions for pos in block_positions):
                    return False
        
        # Verificar conectividade        
        if not self._is_connected(state):
            return False
        
        # Verificar se não há peças iguais adjacentes
        for row in range(self.board.rows):
            for col in range(self.board.cols):
                if self.board.is_position_filled(state, row, col):
                    piece_type = state.filled_grid[row][col]
                    
                    # Usando orientations (EAST, NORTH, WEST, SOUTH) do utils.py
                    for dr, dc in orientations:
                        adj_row, adj_col = row + dr, col + dc
                        
                        if (self.board.is_valid_position(adj_row, adj_col) and
                            self.board.is_position_filled(state, adj_row, adj_col) and
                            state.filled_grid[adj_row][adj_col] == piece_type):
                            return False
        
        return True

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        state = node.state
        h_value = 0
        
        # Heurística baseada no número de opções para regiões não preenchidas
        for region in self.board.get_all_regions():
            if region not in state.placed_pieces:
                possible_placements = 0
                
                for tetromino_type, variants in ALL_TETROMINO_VARIANTS.items():
                    for variant in variants:
                        if self._can_place_shape_in_region(state, region, variant, tetromino_type):
                            possible_placements += 1
                            break
                
                # Penalizar regiões com poucas opções
                if possible_placements == 0:
                    return float('inf')  # Estado inválido
                else:
                    h_value += 1.0 / possible_placements
        
        # Penalização adicional por riscos de desconectividade
        filled_positions = self.board.get_filled_positions(state)
        if filled_positions and not self._is_connected(state):
            h_value += 10
        
        # Penalização por regiões com poucas células (mais difíceis de preencher)
        for region in self.board.get_all_regions():
            if region not in state.placed_pieces:
                region_size = self.board.get_region_size(region)
                if region_size == 4:  # Tamanho mínimo, mais restritivo
                    h_value += 0.5
        
        return h_value
