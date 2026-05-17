import json
import matplotlib.pyplot as plt

def plot_distribuicao_scores():
    # Carrega os dados
    with open('scores_distribuicao_TFIDF.json', 'r') as f:
        scores_tfidf = json.load(f)
        
    with open('scores_distribuicao_BM25.json', 'r') as f:
        scores_bm25 = json.load(f)

    # Configuração do gráfico
    plt.figure(figsize=(8, 6))
    
    # Cria o boxplot
    box = plt.boxplot([scores_tfidf, scores_bm25], 
                      labels=['TF-IDF', 'BM25'], 
                      patch_artist=True,
                      medianprops=dict(color="red", linewidth=2))

    # Cores das caixas
    colors = ['#lightblue', '#lightgreen']
    for patch, color in zip(box['boxes'], ['lightblue', 'lightgreen']):
        patch.set_facecolor(color)

    # Títulos e Eixos
    plt.title('Distribuição de Scores dos Documentos Recuperados', fontsize=14)
    plt.ylabel('Score Absoluto', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Salva em PDF para não perder qualidade no LaTeX
    plt.savefig('scores_boxplot.png', format='png', bbox_inches='tight')
    print("Gráfico 'scores_boxplot.png' gerado com sucesso!")

if __name__ == "__main__":
    plot_distribuicao_scores()