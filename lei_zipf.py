import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_zipf(csv_path, output_image_path):
    print(f"Lendo os dados de {csv_path}...")
    
    try:
        # Lê o arquivo ignorando qualquer cabeçalho que possa (ou não) existir
        # e força os nomes das colunas
        df = pd.read_csv(csv_path, header=None, names=['termo', 'tamanho_lista'], dtype={'tamanho_lista': str})
        
        # Se a primeira linha for o cabeçalho em formato de texto, nós a descartamos
        if df['tamanho_lista'].iloc[0] == 'tamanho_lista':
            df = df.drop(0)
            
        # Converte a coluna para números inteiros (ignorando erros ou sujeiras do CSV)
        df['tamanho_lista'] = pd.to_numeric(df['tamanho_lista'], errors='coerce')
        
        # Remove linhas inválidas (NaN)
        df = df.dropna(subset=['tamanho_lista'])
        
    except Exception as e:
        print(f"Erro ao ler o arquivo: {e}")
        return

    print("Ordenando frequências...")
    # Ordena de forma decrescente
    frequencias = df['tamanho_lista'].sort_values(ascending=False).values
    
    ranks = np.arange(1, len(frequencias) + 1)
    
    # Linha teórica de Zipf
    frequencia_maxima = frequencias[0]
    zipf_teorico = frequencia_maxima / ranks

    print("Gerando o gráfico...")
    plt.figure(figsize=(10, 6))
    
    # Plota os dados reais
    plt.loglog(ranks, frequencias, marker='.', linestyle='none', color='blue', 
               markersize=2, alpha=0.5, label='Coleção Real (Tamanho das Listas)')
    
    # Plota a linha teórica
    plt.loglog(ranks, zipf_teorico, linestyle='--', color='red', 
               linewidth=2, label='Lei de Zipf Teórica')

    plt.title('Distribuição do Tamanho das Listas Invertidas (Lei de Zipf)', fontsize=14)
    plt.xlabel('Rank do Termo (escala logarítmica)', fontsize=12)
    plt.ylabel('Número de Postagens (escala logarítmica)', fontsize=12)
    
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(fontsize=12)
    plt.tight_layout()

    plt.savefig(output_image_path, format='png', dpi=300)
    print(f"Gráfico salvo com sucesso em: {output_image_path}")
    
    plt.show()

if __name__ == "__main__":
    caminho_entrada = "./results/distribuicao_tamanhos.csv"
    caminho_saida = "zipf_plot.png"
    plot_zipf(caminho_entrada, caminho_saida)