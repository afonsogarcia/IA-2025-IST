# nuruomino.py: Template para implementaÃ§Ã£o do projeto de InteligÃªncia Artificial 2024/2025.
# Devem alterar as classes e funÃ§Ãµes neste ficheiro de acordo com as instruÃ§Ãµes do enunciado.
# AlÃ©m das funÃ§Ãµes e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 00:
# 00000 Nome1
# 00000 Nome2

from search import Problem, Node
from sys import stdin
from utils import *  # Importamos todas as funÃ§Ãµes disponÃ­veis em utils.py

class NuruominoState:
    state_id = 0

    def __init__(self, board, filled_grid=None, placed_pieces=None):
        self.board = board  # Board original com regiÃµes
        self.id = NuruominoState.state_id
        NuruominoState.state_id += 1
        
        # Grid com as peÃ§as colocadas (None = vazio, 'L'/'I'/'T'/'S' = peÃ§a)
        if filled_grid is None:
            self.filled_grid = [[None for _ in range(board.cols)] for _ in range(board.rows)]
        else:
            self.filled_grid = [row[:] for row in filled_grid]  # Deep copy
          # DicionÃ¡rio das peÃ§as colocadas por regiÃ£o {region_id: tetromino_type}
        self.placed_pieces = placed_pieces.copy() if placed_pieces else {}

    def __lt__(self, other):
        """ Este mÃ©todo Ã© utilizado em caso de empate na gestÃ£o da lista
        de abertos nas procuras informadas. """
        return self.id < other.id
    
    def clone(self):
        """Cria uma cÃ³pia profunda do estado."""
        return NuruominoState(self.board, self.filled_grid, self.placed_pieces)
    
    def is_position_filled(self, row, col):
        """Verifica se uma posiÃ§Ã£o estÃ¡ preenchida."""
        return self.filled_grid[row][col] is not None
    
    def get_filled_positions(self):
        """Retorna todas as posiÃ§Ãµes preenchidas."""
        filled = []
        for row in range(self.board.rows):
            for col in range(self.board.cols):
                if self.is_position_filled(row, col):
                    filled.append((row, col))
        return filled

    def get_value(self, row: int, col: int):
        """Retorna o valor numa determinada posição, considerando peças colocadas."""
        # Ajustar coordenadas para 0-based se necessário
        adj_row, adj_col = row - 1, col - 1
        
        if 0 <= adj_row < self.board.rows and 0 <= adj_col < self.board.cols:
            # Se há uma peça colocada nesta posição, retornar o tipo da peça
            if self.is_position_filled(adj_row, adj_col):
                return self.filled_grid[adj_row][adj_col]
            else:
                # Se não há peça, retornar o valor original da região
                return self.board.grid[adj_row][adj_col]
        return None
    
    def adjacent_values(self, row: int, col: int) -> list:
        """Retorna os valores das células adjacentes à posição, considerando peças colocadas."""
        values = []
        
        # Converter coordenadas baseadas em 1 para baseadas em 0
        adj_row, adj_col = row - 1, col - 1
        
        for pos_row, pos_col in self.board.adjacent_positions(adj_row, adj_col):
            # Se a posição tem uma peça colocada, retornar o tipo da peça
            if self.is_position_filled(pos_row, pos_col):
                values.append(self.filled_grid[pos_row][pos_col])
            else:
                # Se não tem peça, retornar o valor original da região
                values.append(self.board.grid[pos_row][pos_col])
        
        return values

    def print_board(self):
        """Imprime o tabuleiro atual mostrando as peças colocadas."""
        print("Board atual com peças colocadas:")
        for row in range(self.board.rows):
            for col in range(self.board.cols):
                if self.is_position_filled(row, col):
                    # Mostrar o tipo da peça (L, I, T, S)
                    print(self.filled_grid[row][col], end='\t')
                else:
                    # Mostrar o número da região original
                    print(self.board.grid[row][col], end='\t')
            print()  # Nova linha no final de cada row
        print()  # Linha em branco no final

class Board:
    """RepresentaÃ§Ã£o interna de um tabuleiro do Puzzle Nuruomino."""
    
    def __init__(self, grid):
        """Inicializa o tabuleiro com a grelha de regiÃµes."""
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if grid else 0
        # Criar um mapeamento de regiÃµes para posiÃ§Ãµes
        self.regions = {}
        for row in range(self.rows):
            for col in range(self.cols):
                region = self.grid[row][col]
                if region not in self.regions:
                    self.regions[region] = []
                self.regions[region].append((row, col))
    
    def get_value(self, row: int, col: int):
        """Retorna o valor numa determinada posição."""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.grid[row-1][col-1]
        return None
    
    def adjacent_regions(self, region: int) -> list:
        """Devolve uma lista das regiÃµes que fazem fronteira com a regiÃ£o enviada no argumento."""
        adjacent = set()
        
        # Para cada posiÃ§Ã£o da regiÃ£o
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
        """Devolve as posiÃ§Ãµes adjacentes Ã  posiÃ§Ã£o, em todas as direÃ§Ãµes, incluindo diagonais."""
        positions = []
        
        # Verificar todas as 8 direÃ§Ãµes (incluindo diagonais)
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:  # Pular a prÃ³pria posiÃ§Ã£o
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
        """Retorna todas as posiÃ§Ãµes que pertencem a uma regiÃ£o."""
        return self.regions.get(region, [])
    
    def get_all_regions(self) -> list:
        """Retorna uma lista com todos os identificadores de regiÃµes."""
        return list(self.regions.keys())
    def print_board(self):
        """Imprime o tabuleiro no formato original."""
        # Usando a funÃ§Ã£o print_table do utils.py
        print_table(self.grid, sep='\t')
    
    @staticmethod
    def parse_instance():
        """LÃª a instÃ¢ncia do problema do standard input (stdin)
        e retorna uma instÃ¢ncia da classe Board.

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
        """LÃª a instÃ¢ncia do problema de um arquivo
        e retorna uma instÃ¢ncia da classe Board.
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
            print(f"Erro: Arquivo {filename} nÃ£o encontrado.")
            return None
        except Exception as e:
            print(f"Erro ao ler arquivo {filename}: {e}")
            return None
        
        return Board(grid)

    def is_valid_position(self, row: int, col: int) -> bool:
        """Verifica se uma posiÃ§Ã£o Ã© vÃ¡lida no tabuleiro."""
        return 0 <= row < self.rows and 0 <= col < self.cols
    
    def get_region_size(self, region: int) -> int:
        """Retorna o nÃºmero de cÃ©lulas numa regiÃ£o."""
        return len(self.regions.get(region, []))
    
    def get_region_for_position(self, row: int, col: int) -> int:
        """Retorna o identificador da regiÃ£o que contÃ©m a posiÃ§Ã£o (row, col)."""
        if self.is_valid_position(row, col):
            return self.grid[row][col]
        return None
    def get_orthogonal_neighbors(self, row: int, col: int) -> list:
        """Retorna as posiÃ§Ãµes ortogonalmente adjacentes (sem diagonais)."""
        neighbors = []
        # Usando orientations do utils.py (EAST, NORTH, WEST, SOUTH)
        for dr, dc in orientations:
            new_row, new_col = row + dr, col + dc
            if self.is_valid_position(new_row, new_col):
                neighbors.append((new_row, new_col))
        return neighbors
    
    def get_orthogonal_values(self, row: int, col: int) -> list:
        """Retorna os valores das cÃ©lulas ortogonalmente adjacentes."""
        values = []
        for pos_row, pos_col in self.get_orthogonal_neighbors(row, col):
            values.append(self.grid[pos_row][pos_col])
        return values

# DefiniÃ§Ãµes dos tetraminÃ³s (formas bÃ¡sicas)
TETROMINOS = {
    'L': [(0, 0), (1, 0), (2, 0), (2, 1)],
    'I': [(0, 0), (1, 0), (2, 0), (3, 0)],
    'T': [(0, 0), (0, 1), (0, 2), (1, 1)],
    'S': [(0, 0), (0, 1), (1, 1), (1, 2)]
}

def rotate_90(shape):
    """Roda uma forma 90 graus no sentido horÃ¡rio."""
    return [(y, -x) for x, y in shape]

def reflect_horizontal(shape):
    """Reflete uma forma horizontalmente."""
    return [(-x, y) for x, y in shape]

def normalize_shape(shape):
    """Normaliza uma forma para que comece em (0, 0)."""
    # Usando a funÃ§Ã£o utilitÃ¡ria min de forma mais eficiente
    xs = [x for x, y in shape]
    ys = [y for x, y in shape]
    min_x = min(xs)
    min_y = min(ys)
    return [(x - min_x, y - min_y) for x, y in shape]

def generate_all_variants(base_shape):
    """Gera todas as variantes (rotaÃ§Ãµes + reflexÃµes) de uma forma."""
    variants = set()
    current = base_shape
    
    # 4 rotaÃ§Ãµes
    for _ in range(4):
        variants.add(tuple(sorted(normalize_shape(current))))
        # Adicionar reflexÃ£o de cada rotaÃ§Ã£o
        reflected = reflect_horizontal(current)
        variants.add(tuple(sorted(normalize_shape(reflected))))
        current = rotate_90(current)
    
    return [list(variant) for variant in variants]

# Gerar todas as variantes de cada tetraminÃ³
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
        """Verifica se uma forma pode ser colocada numa regiÃ£o especÃ­fica."""
        region_positions = set(self.board.get_region_positions(region))
        
        # O tetromino deve caber dentro da regiÃ£o, mas nÃ£o precisa preenchÃª-la completamente
        if len(shape) > len(region_positions):
            return None
        
        # Tentar colocar a forma em cada posiÃ§Ã£o possÃ­vel da regiÃ£o
        for start_row, start_col in region_positions:
            shape_positions = []
            valid_placement = True
            
            for dx, dy in shape:
                new_row, new_col = start_row + dx, start_col + dy
                shape_positions.append((new_row, new_col))
                
                # Verificar se a posiÃ§Ã£o estÃ¡ dentro da regiÃ£o
                if (new_row, new_col) not in region_positions:
                    valid_placement = False
                    break
                
                # Verificar se a posiÃ§Ã£o jÃ¡ estÃ¡ ocupada
                if state.is_position_filled(new_row, new_col):
                    valid_placement = False
                    break
            
            if valid_placement and len(set(shape_positions)) == len(shape):
                # Verificar se nÃ£o cria blocos 2x2
                if not self._would_create_2x2_block(state, shape_positions, tetromino_type):                    # Verificar se nÃ£o hÃ¡ peÃ§as iguais adjacentes
                    if not self._would_create_adjacent_same_pieces(state, shape_positions, tetromino_type):
                        return shape_positions
        
        return None
    
    def _would_create_2x2_block(self, state, new_positions, tetromino_type):
        """Verifica se colocar as peÃ§as nas posiÃ§Ãµes criaria um bloco 2x2."""
        # Criar um estado temporÃ¡rio com as novas posiÃ§Ãµes
        temp_filled = set(state.get_filled_positions())
        temp_filled.update(new_positions)
        
        # Verificar todos os possÃ­veis blocos 2x2
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
        """Verifica se a colocaÃ§Ã£o criaria peÃ§as iguais ortogonalmente adjacentes."""
        for row, col in new_positions:
            # Verificar vizinhos ortogonais usando orientations do utils
            for dr, dc in orientations:
                adj_row, adj_col = row + dr, col + dc
                
                if (self.board.is_valid_position(adj_row, adj_col) and 
                    state.is_position_filled(adj_row, adj_col)):
                      # Verificar se Ã© uma peÃ§a do mesmo tipo
                    if state.filled_grid[adj_row][adj_col] == tetromino_type:
                        return True
        
        return False

    def _is_connected(self, state):
        """Verifica se todas as peÃ§as preenchidas estÃ£o ortogonalmente conectadas."""
        filled_positions = set(state.get_filled_positions())
        
        if not filled_positions:
            return True
        
        # BFS para verificar conectividade usando funÃ§Ãµes do utils
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
        """Retorna regiÃµes ainda nÃ£o preenchidas, ordenadas por MRV."""
        available_regions = []
        
        for region in self.board.get_all_regions():
            if region not in state.placed_pieces:
                # Contar quantas formas podem ser colocadas nesta regiÃ£o
                possible_placements = 0
                
                for tetromino_type, variants in ALL_TETROMINO_VARIANTS.items():
                    # Verificar se pelo menos uma variante deste tipo pode ser colocada
                    can_place_type = False
                    for variant in variants:
                        if self._can_place_shape_in_region(state, region, variant, tetromino_type):
                            can_place_type = True
                            break  # Uma variante Ã© suficiente para este tipo
                    
                    if can_place_type:
                        possible_placements += 1
                
                available_regions.append((region, possible_placements))
        
        # Ordenar por MRV (menor nÃºmero de opÃ§Ãµes primeiro)
        available_regions.sort(key=lambda x: x[1])
        return [region for region, _ in available_regions]

    def actions(self, state):
        """Retorna uma lista de aÃ§Ãµes que podem ser executadas a
        partir do estado passado como argumento."""
        actions = []
        available_regions = self._get_available_regions(state)
        
        # Se nÃ£o hÃ¡ regiÃµes disponÃ­veis, nÃ£o hÃ¡ aÃ§Ãµes
        if not available_regions:
            return actions
        
        # Aplicar MRV: focar na regiÃ£o com menos opÃ§Ãµes
        target_region = available_regions[0]
          # Tentar todas as peÃ§as e variantes possÃ­veis para esta regiÃ£o        
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
        """Retorna True se e sÃ³ se o estado passado como argumento Ã©
        um estado objetivo."""
        # Verificar se todas as regiÃµes estÃ£o preenchidas
        if len(state.placed_pieces) != len(self.board.get_all_regions()):
            return False
        
        # Verificar se nÃ£o hÃ¡ blocos 2x2
        filled_positions = set(state.get_filled_positions())
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
        
        # Verificar se nÃ£o hÃ¡ peÃ§as iguais adjacentes
        for row in range(self.board.rows):
            for col in range(self.board.cols):
                if state.is_position_filled(row, col):
                    piece_type = state.filled_grid[row][col]
                    
                    # Usando orientations (EAST, NORTH, WEST, SOUTH) do utils.py
                    for dr, dc in orientations:
                        adj_row, adj_col = row + dr, col + dc
                        
                        if (self.board.is_valid_position(adj_row, adj_col) and
                            state.is_position_filled(adj_row, adj_col) and
                            state.filled_grid[adj_row][adj_col] == piece_type):
                            return False
        
        return True

    def h(self, node: Node):
        """FunÃ§Ã£o heuristica utilizada para a procura A*."""
        state = node.state
        h_value = 0
        
        # HeurÃ­stica baseada no nÃºmero de opÃ§Ãµes para regiÃµes nÃ£o preenchidas
        for region in self.board.get_all_regions():
            if region not in state.placed_pieces:
                possible_placements = 0
                
                for tetromino_type, variants in ALL_TETROMINO_VARIANTS.items():
                    for variant in variants:
                        if self._can_place_shape_in_region(state, region, variant, tetromino_type):
                            possible_placements += 1
                            break
                
                # Penalizar regiÃµes com poucas opÃ§Ãµes
                if possible_placements == 0:
                    return float('inf')  # Estado invÃ¡lido
                else:
                    h_value += 1.0 / possible_placements
        
        # PenalizaÃ§Ã£o adicional por riscos de desconectividade
        filled_positions = state.get_filled_positions()
        if filled_positions and not self._is_connected(state):
            h_value += 10
        
        # PenalizaÃ§Ã£o por regiÃµes com poucas cÃ©lulas (mais difÃ­ceis de preencher)
        for region in self.board.get_all_regions():
            if region not in state.placed_pieces:
                region_size = self.board.get_region_size(region)
                if region_size == 4:  # Tamanho mÃ­nimo, mais restritivo
                    h_value += 0.5
        
        return h_value
