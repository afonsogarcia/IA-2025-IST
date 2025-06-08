# live_visualizer.py: Visualizador em tempo real da evolução do puzzle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import os

class LiveVisualizer:
    def __init__(self, board, delay=0.5, save_steps=False):
        self.initial_board = board
        self.delay = delay
        self.save_steps = save_steps
        self.step_count = 0
        self.fig = None
        self.ax = None
        
        # Cores para cada tipo de peça
        self.colors = {
            'L': '#FF6B6B',    # Vermelho
            'I': '#4ECDC4',    # Azul claro  
            'T': '#45B7D1',    # Azul
            'S': '#96CEB4',    # Verde
            'O': '#FFEAA7',    # Amarelo
            'empty': '#F8F9FA',  # Branco
            'region': '#E0E0E0'  # Cinza claro
        }
        
        # Configurar matplotlib para modo interativo
        plt.ion()
        
        if save_steps:
            self.output_dir = "evolution_steps"
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
    
    def setup_plot(self, board):
        """Configura a janela de visualização."""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(board.cols * 0.7, board.rows * 0.7))
            
        self.ax.clear()
        self.ax.set_xlim(0, board.cols)
        self.ax.set_ylim(0, board.rows)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
    
    def draw_board(self, board, action=None):
        """Desenha o estado atual do tabuleiro."""
        self.setup_plot(board)
        
        # Desenhar cada célula
        for row in range(board.rows):
            for col in range(board.cols):
                if board.is_position_filled(row, col):
                    piece_type = board.get_filled_value(row, col)
                    color = self.colors.get(piece_type, '#CCCCCC')
                    text = piece_type
                    text_color = 'white'
                    text_weight = 'bold'
                else:
                    color = self.colors['region']
                    text = str(board.get_value(row, col))
                    text_color = 'black'
                    text_weight = 'normal'
                
                # Criar retângulo
                rect = patches.Rectangle((col, board.rows - row - 1), 1, 1,
                                       linewidth=2, edgecolor='black',
                                       facecolor=color)
                self.ax.add_patch(rect)
                
                # Adicionar texto
                self.ax.text(col + 0.5, board.rows - row - 0.5, text,
                           ha='center', va='center', fontsize=12,
                           color=text_color, weight=text_weight)
        
        # Título com informações
        pieces_placed = len(board.placed_pieces)
        total_regions = len(board.get_all_regions())
        title = f'Nuruomino - Passo {self.step_count} - Peças: {pieces_placed}/{total_regions}'
        
        if action:
            region_id, piece_type = action[0], action[1]
            title += f' | Última: {piece_type} na região {region_id}'
        
        self.ax.set_title(title, fontsize=14, pad=20)
        
        # Atualizar display
        plt.draw()
        plt.pause(0.01)  # Pequena pausa para renderização
        
        # Salvar passo se solicitado
        if self.save_steps:
            filename = os.path.join(self.output_dir, f"step_{self.step_count:03d}.png")
            self.fig.savefig(filename, bbox_inches='tight', dpi=150)
    
    def show_step(self, board, action=None):
        """Mostra um passo da evolução com delay."""
        self.step_count += 1
        
        print(f"Passo {self.step_count}: ", end="")
        if action:
            region_id, piece_type = action[0], action[1]
            print(f"Colocando peça '{piece_type}' na região {region_id}")
        else:
            print("Estado inicial")
        
        self.draw_board(board, action)
        
        # Aguardar delay se não for o primeiro passo
        if self.step_count > 1:
            time.sleep(self.delay)
    
    def show_final(self, board):
        """Mostra o resultado final."""
        self.draw_board(board)
        
        # Destacar que é a solução final
        current_title = self.ax.get_title()
        self.ax.set_title(current_title + " | SOLUÇÃO ENCONTRADA!", 
                         fontsize=16, pad=20, color='green', weight='bold')
        
        plt.draw()
        
        print("\n🎉 PUZZLE RESOLVIDO! 🎉")
        print(f"Total de passos: {self.step_count}")
        
        if self.save_steps:
            print(f"Imagens salvas em: {self.output_dir}/")
    
    def show_final_statistics(self):
        """Mostra estatísticas finais da solução."""
        print(f"📈 Estatísticas da visualização:")
        print(f"   Passos visualizados: {self.step_count}")
        if self.save_steps:
            print(f"   Imagens salvas em: visualizations/")
        print(f"   Delay entre passos: {self.delay}s")
        
        # Manter a última visualização aberta
        if self.fig:
            plt.show(block=True)
    
    def close(self):
        """Fecha a visualização."""
        if self.fig:
            plt.ioff()
            plt.show()  # Manter janela aberta até o usuário fechar
    
    def create_gif(self, output_file="nuruomino_solution.gif", duration=1000):
        """Cria um GIF animado dos passos salvos."""
        if not self.save_steps:
            print("Para criar GIF, execute com save_steps=True")
            return
        
        try:
            from PIL import Image
            import glob
            
            # Encontrar todas as imagens
            image_files = sorted(glob.glob(os.path.join(self.output_dir, "step_*.png")))
            
            if not image_files:
                print("Nenhuma imagem encontrada para criar GIF")
                return
            
            # Carregar imagens
            frames = []
            for image_file in image_files:
                img = Image.open(image_file)
                frames.append(img)
            
            # Salvar como GIF
            frames[0].save(output_file, save_all=True, append_images=frames[1:], 
                          duration=duration, loop=0)
            print(f"GIF criado: {output_file}")
            
        except ImportError:
            print("Para criar GIFs, instale Pillow: pip install Pillow")
        except Exception as e:
            print(f"Erro ao criar GIF: {e}")

# Função auxiliar para uso rápido
def create_live_visualizer(board, delay=0.5, save_steps=False):
    """Cria um visualizador em tempo real."""
    return LiveVisualizer(board, delay, save_steps)
