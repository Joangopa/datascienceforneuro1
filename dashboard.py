import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import joblib

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from scipy.stats import shapiro, ttest_ind, mannwhitneyu





# Configuração inicial da página
st.set_page_config(
    page_title="Meu Dashboard Analítico",
    page_icon="📊",
    layout="wide"
)

st.markdown(
    """
    <style>
        .stApp {
            background-color: #e6f2ff;  /* Cor de fundo cinza claro */
        }
    </style>
    """, 
    unsafe_allow_html=True
)

# Carregar os dados
@st.cache_data
def load_data():
    data = pd.read_csv('arquivos/oasis_cross-sectional.csv')
    data = data.dropna(subset=['MMSE', 'CDR'])
    data.drop('Delay', axis=1, inplace=True)
    return data

data = load_data()

@st.cache_data
def load_data2():
    data = pd.read_csv('arquivos/oasis_cross-sectional.csv')
    # Filtrar os dados onde MMSE e CDR não são nulos
    data = data.dropna(subset=['MMSE', 'CDR'])
    data.drop('Delay', axis=1, inplace=True)
    data["CDR"] = data["CDR"].astype(str)

    return data

data2 = load_data2()








# Criando as abas
tab_title, tab_intro_problema,  tab_estudo, tab_analises, tab_pred, tab_pca, tab_simulacao, tab_conclusoes = st.tabs(["-", "📌 Introdução ao Problema", "Estudo",  
                                                   "📈 Análises",
                                                  "Predições", "PCA e Agrupamento", "Simulações",
                                                  "Conclusões"])

with tab_title:
    st.markdown(
        """
        <h1 style="text-align: center; font-size: 40px; color: #0073e6;">
            Análise de Alzheimer: Diagnóstico e Tendências
        </h1>
        """,
        unsafe_allow_html=True
    )
    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        st.image("image_title.webp", use_container_width =False,  width=400)


with tab_intro_problema:
    
    col1, col2 = st.columns([1, 1])

    with col1:
        
        # st.subheader("Introdução ao Problema")
        with st.container(border=True):  
            sinais = [
            "📉 **Perda de memória** – Esquecimento frequente de informações recentes.",
            "👜 **Perder pertences ou deixar objetos em lugares inusitados** – Como colocar chaves na geladeira.",
            "🛠️ **Dificuldade em realizar tarefas cotidianas** – Como cozinhar, dirigir ou pagar contas.",
            "🧭 **Desorientação no espaço e no tempo** – Perder-se em lugares conhecidos.",
            "🧩 **Dificuldade no planejamento e resolução de problemas** – Como seguir uma receita simples.",
            "⚖️ **Tomada de decisões inadequadas** – Como confiar em pessoas erradas ou gastar muito dinheiro.",
            "🗣️ **Dificuldade de expressar e compreender a língua** – Esquecer palavras ou repetir frases.",
            "👀 **Problemas de atenção, concentração e percepção** – Dificuldade para manter o foco."
        ]

        for sinal in sinais:
            st.markdown(f"- {sinal}")

           
    with col2:  
        # Você pode adicionar imagens, gráficos ou outros elementos
        
        st.image("brain.png", caption="Atrofia Cerebral", width=400)


with tab_estudo: 
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Descrição dos dados:")
        with st.container(border=True):
            st.write("""
                    - 416 pessoas participaram do estudo;
                    - Idades entre 18 e 96 anos;
                    - Para cada pessoa, são incluídas dados de ressonâncias magnéticas individuais, obtidas em sessões de varredura única;
                    - Todos destros;
                    - Inclui homens e mulheres;
                    - Um conjunto de dados de confiabilidade é incluído contendo 20 sujeitos não dementes fotografados em uma visita subsequente dentro de 90 dias de sua sessão inicial.
            """)
        st.subheader("Tratamento dos dados:")
        with st.container(border=True):
            st.write("""
                    - Os dados foram filtrados para incluir apenas aqueles com pontuação MMSE e CDR não nulas;
                    - As colunas de dados irrelevantes foram removidas;
                    - Dos 416 dados originais, 235 foram mantidos após o pré-processamento.
                    - Os dados que foram removidos pertencem a pessoas com idades abaixo de 59 anos
            """)
    
    with col2:
        st.subheader("📊 Variáveis do Conjunto de Dados")
        
        # Container para as variáveis
        with st.container(border=True):
            variaveis = {
                "ID": "Identificação",
                "M/F": "M = Masculino, F = Feminino",
                "Mão": "Mão Dominante",
                "Idade": "Idade em anos",
                "Educ": "Nível de Educação, de 1 a 5",
                "NSE": "Nível Socioeconômico, de 1 a 5",
                "eTIV": "Volume Intracraniano Total Estimado",
                "ASF": "Fator de Escala Atlas",
                "nWBV": "Volume Cerebral Total Normalizado",
                "MMSE": "Mini Exame do Estado Mental – escala de 0 a 30",
                "CDR": "Escala Clínica de Demência. 0 = Sem Demência, 0.5 = Demência Muito Leve, 1 = Demência Leve, 2 = Demência Moderada"
            }

            for var, desc in variaveis.items():
                st.markdown(f"**{var}**: {desc}")
        
        # Botão para mostrar/ocultar a imagem
        if st.button('MMSE'):
            if 'show_image' not in st.session_state:
                st.session_state.show_image = False
            st.session_state.show_image = not st.session_state.show_image
        
        if st.session_state.get('show_image', False):
            st.image("mmse.jpg", use_container_width =True)

with tab_analises:
    
    # st.header("Análise de Correlação")

    subtab1, subtab2, subtab3, subtab4 = st.tabs(["Distribuição de CDR", "Análise de Correlação", "nWBV vs CDR", "MMSE vs CDR"])
    
    

    with subtab1:

        col1, col2 = st.columns([2, 1])
        
        with col1:
            cdr_table = data.groupby(['CDR']).size().reset_index(name='Count')
            
            cdr_descricao = {
                0.0: 'Sem demência',
                0.5: 'Demência muito leve',
                1.0: 'Demência leve',
                2.0: 'Demência moderada'
            }

            # Substituir os valores da coluna CDR
            cdr_table['Interpretação'] = cdr_table['CDR'].map(cdr_descricao)
            cdr_table = cdr_table[['CDR','Interpretação','Count']]

            # Definir a paleta de cores personalizada
            cores_personalizadas = {
                0.0: '#4daf4a',  # Verde intermediário
                0.5: '#ff9999',  # Vermelho leve
                1.0: '#e41a1c',  # Vermelho intermediário
                2.0: '#990000'   # Vermelho intenso
            }

            # Mapear as cores para cada interpretação
            cdr_table['Cor'] = cdr_table['CDR'].map(cores_personalizadas)

            plt.figure(figsize=(4, 2))
            ax = sns.barplot(
                x='Count', 
                y='Interpretação', 
                data=cdr_table, 
                hue='Interpretação', 
                palette=cdr_table['Cor'].tolist(),  # Usar a lista de cores personalizadas
                dodge=False
            )

            # Adicionando os valores de contagem no final de cada barra
            for index, row in cdr_table.iterrows():
                ax.text(row['Count'] + 1, index, str(row['Count']), color='black', va='center')

            # Remover as bordas do gráfico
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Adicionando títulos e rótulos
            plt.title('Distribuição de Casos por Tipo de Demência')
            plt.xlabel('Número de Casos')
            plt.ylabel('Tipo de Demência')

            # Remover a legenda de cores (opcional, já que os rótulos estão no eixo Y)
            # ax.legend_.remove()

            # Exibindo o gráfico no Streamlit
            st.pyplot(plt, use_container_width=False)
        
        with col2:
            st.markdown("""
            - 100 dos sujeitos incluídos com mais de 60 anos foram clinicamente diagnosticados com doença de Alzheimer muito leve a moderada.
                        """)
        

    with subtab2:
    
        # Criar duas colunas (1:2 - a figura ocupará 1/3 do espaço)
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Cria a figura menor
            fig, ax = plt.subplots(figsize=(5, 3))  # Tamanho reduzido
            corr_spearman = data.corr(method='spearman', numeric_only=True)
            mask = np.triu(np.ones_like(corr_spearman, dtype=bool))
            
            sns.heatmap(
                corr_spearman,
                mask=mask,
                annot=True,
                fmt=".2f",
                cmap='coolwarm',
                center=0,
                vmin=-1,
                vmax=1,
                cbar_kws={"shrink": 0.6},
                ax=ax,
                annot_kws={"size": 6} 
            )
            
            # Ajustar fonte da colorbar
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=6) 

            ax.set_title("Correlação (Spearman)", fontsize=10)
            plt.xticks(rotation=45, fontsize=6)
            plt.yticks(rotation=0, fontsize=6)
            plt.tight_layout()
            
            st.pyplot(fig, use_container_width=False)
        
        with col2:
            st.markdown("""
            **Mapa de Correlação de Spearman**
            
            Este gráfico mostra as relações entre as variáveis numéricas:
            
            - **Correlação Positiva (+1)**
            - **Correlação Negativa (-1)**
            - **Sem Correlação (0)**
            """)

    with subtab3:

        st.title("Análise Estatística de nWBV entre Pacientes com e sem Demência")

        fig2 = px.scatter(
            data2,
            x='Age',
            y='nWBV',
            color='CDR',
            size='eTIV',
            hover_name='ID',
            title='Volume Cerebral Normalizado por Idade e CDR',
            labels={'Age': 'Idade', 'nWBV': 'Volume Cerebral Normalizado', 'CDR': 'CDR'},
            color_discrete_sequence=['#4daf4a', '#ff9999', '#e41a1c', '#990000']  
        )
        st.plotly_chart(fig2, use_container_width=True)

        nwbv_doentes_maiores_60 = data.loc[(data['Age'] > 60) & (data['CDR'] >0), ['nWBV']].reset_index(drop=True)
        nwbv_nao_doentes_maiores_60 = data.loc[(data['Age'] > 60) & (data['CDR']  == 0), ['nWBV']]

        def cohens_d(x, y):
            nx = len(x)
            ny = len(y)
            dof = nx + ny - 2
            pooled_std = np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)
            return (np.mean(x) - np.mean(y)) / pooled_std
    
        st.header("Distribuição do Volume Cerebral Total Normalizado")
    
        # Definir paleta de cores
        cor_nao_doentes = '#4daf4a'  # Verde
        cor_doentes = '#ff7f00'      # Laranja intenso

        # Criar 3 colunas (a terceira terá o dobro do tamanho)
        col1, col2, col3 = st.columns([1, 1, 1.5])

        # Gráficos de distribuição (colunas 1 e 2)
        with col1:
            #st.subheader(" ")
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.histplot(nwbv_doentes_maiores_60['nWBV'], kde=True, color=cor_doentes, label='CDR > 0')
            plt.title("Distribuição - Doentes (CDR > 0)")
            plt.xlabel("nWBV")
            plt.legend()
            st.pyplot(fig)

        with col2:
            #st.subheader(" ")
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.histplot(nwbv_nao_doentes_maiores_60['nWBV'], kde=True, color=cor_nao_doentes, label='CDR = 0')
            plt.title("Distribuição - Não Doentes (CDR = 0)")
            plt.xlabel("nWBV")
            plt.legend()
            st.pyplot(fig)

        # Boxplot (coluna 3 - mais larga)
        with col3:
            
            fig, ax = plt.subplots(figsize=(8, 4))
            
            # Preparar os dados
            nwbv_doentes_maiores_60['Grupo'] = 'CDR > 0'
            nwbv_nao_doentes_maiores_60['Grupo'] = 'CDR = 0'
            dados_comparacao = pd.concat([nwbv_doentes_maiores_60, nwbv_nao_doentes_maiores_60])
            
            # Criar o boxplot com a nova paleta
            sns.boxplot(x='Grupo', y='nWBV', data=dados_comparacao,
                        hue='Grupo', 
                        palette={'CDR > 0': cor_doentes, 'CDR = 0': cor_nao_doentes},
                        order=['CDR > 0', 'CDR = 0'],
                        legend=False)
            
            plt.title("Comparação de nWBV: Doentes vs Não Doentes")
            plt.ylabel("nWBV")
            plt.xlabel("")
            st.pyplot(fig)


        # Seção 2: Testes de Normalidade
        st.header("Testes de Normalidade (Shapiro-Wilk)")

        # Container com largura reduzida para o slider
        with st.container():
            col_slider, _ = st.columns([0.4, 0.6])  # Slider ocupará apenas 40% da largura
            
            with col_slider:
                alpha1 = st.slider("Nível de significância (α)", 
                                min_value=0.01, 
                                max_value=0.10, 
                                value=0.05, 
                                step=0.01,
                                help="Nível de significância para os testes estatísticos",
                                key="alpha_nwbv")

        # Realizar os testes
        stat_doentes, p_doentes = shapiro(nwbv_doentes_maiores_60['nWBV'])
        stat_nao_doentes, p_nao_doentes = shapiro(nwbv_nao_doentes_maiores_60['nWBV'])

        # Exibir resultados em colunas
        norm_col1, norm_col2 = st.columns(2)

        with norm_col1:
            # Card expandido para Doentes
            with st.expander("**Doentes (CDR > 0)**", expanded=True):
                st.markdown(f"""
                - **Estatística do teste:** {stat_doentes:.4f}
                - **Valor-p:** {p_doentes:.4f}
                - **α selecionado:** {alpha1}
                """)
                st.markdown(f"#### Conclusão: {'Normal' if p_doentes > alpha1 else 'Não normal'}")

        with norm_col2:
            # Card expandido para Não Doentes
            with st.expander("**Não Doentes (CDR = 0)**", expanded=True):
                st.markdown(f"""
                - **Estatística do teste:** {stat_nao_doentes:.4f}
                - **Valor-p:** {p_nao_doentes:.4f}
                - **α selecionado:** {alpha1}
                """)
                st.markdown(f"#### Conclusão: {'Normal' if p_nao_doentes > alpha1 else 'Não normal'}")
        
        # Seção 3: Teste T e Tamanho do Efeito
        st.header("Comparação entre Grupos")

        # Cálculos estatísticos
        t_stat, p_valor = ttest_ind(
            nwbv_doentes_maiores_60['nWBV'],
            nwbv_nao_doentes_maiores_60['nWBV'],
            alternative='less'
        )

        d = cohens_d(nwbv_doentes_maiores_60['nWBV'], nwbv_nao_doentes_maiores_60['nWBV'])

        # Layout em duas colunas
        col_t, col_d = st.columns(2)

        # Coluna 1 - Teste T
        with col_t:
            with st.container(border=True):  # Adiciona um quadro ao redor de col_t
                st.subheader("Teste T para Amostras Independentes")

                with st.expander("Hipóteses", expanded=True):
                    st.markdown("""
                    - **H₀ (Nula):** μ₁ ≥ μ₂ (Doentes têm nWBV maior/igual)
                    - **H₁ (Alternativa):** μ₁ < μ₂ (Doentes têm nWBV menor)
                    """)

                col_esq, col_dir = st.columns(2)

                # Coluna da esquerda
                with col_esq:
                    st.markdown(f"""
                    **Resultados:**
                    - Estatística t = `{t_stat:.4f}`
                    - Graus de liberdade = `{len(nwbv_doentes_maiores_60) + len(nwbv_nao_doentes_maiores_60) - 2}`
                    """)

                # Coluna da direita
                with col_dir:
                    st.metric(label="Valor-p", 
                            value=f"{p_valor:.4f}",
                            delta="Significativo" if p_valor < alpha1 else "Não significativo",
                            delta_color="inverse")

                st.markdown(f"""
                ### Conclusão do Teste T
                {'✅ **Rejeitamos H₀** (p < α)' if p_valor < alpha1 else '❌ **Não rejeitamos H₀** (p ≥ α)'}
                α = {alpha1}
                """)

        with col_d:
            with st.container(border=True):  # Adiciona um quadro ao redor de col_d
                # Criando duas colunas: esquerda para valores e interpretação, direita para tabela de referência
                col_esq, col_dir = st.columns(2)

                # Coluna da esquerda
                with col_esq:
                    st.subheader("Tamanho do Efeito (Cohen's d)")

                    # Aumentando a fonte do valor calculado com HTML
                    st.markdown(f"""
                    <style>
                        .valor-calculado {{
                            font-size: 30px;
                            font-weight: bold;
                            color: #e41a1c;  /* Vermelho intermediário */
                        }}
                    </style>
                    **Valor calculado:**  
                    <p class="valor-calculado">d = {d:.2f}</p>
                    """, unsafe_allow_html=True)

                    # Determinar interpretação
                    abs_d = abs(d)
                    if abs_d >= 0.8:
                        interpretacao = "**Grande efeito** 🟠"
                    elif abs_d >= 0.5:
                        interpretacao = "**Médio efeito** 🟡"
                    else:
                        interpretacao = "**Pequeno efeito** 🟢"

                    st.markdown(f"""
                    ### Interpretação  
                    {interpretacao}  
                    **Direção:** {'Negativo' if d < 0 else 'Positivo'}
                    """)

                # Coluna da direita - Tabela de referência
                with col_dir:
                    st.markdown("""
                    **Escala de referência:**  
                    | d    | Interpretação |
                    |------|---------------|
                    | 0.2  | Pequeno       |
                    | 0.5  | Médio         |
                    | 0.8  | Grande        |
                    """)


    with subtab4:
        st.header("Análise Comparativa Exame Mental entre Pacientes com e sem Demência")
        
        # =============================================
        # Seção 1: Visualização dos Dados
        # =============================================
        
        # Criar duas colunas para os gráficos
        col_graph1, col_graph2 = st.columns(2)
        
        with col_graph1:
            # Gráfico de Strip
            custom_colors = ['#4daf4a', '#ff9999', '#e41a1c', '#990000']   
            fig_strip = px.strip(
                data2,
                y='MMSE',
                x='CDR',
                color='CDR',
                stripmode='overlay',
                title='Distribuição Individual de Pontuações do Exame Mental ',
                labels={'MMSE': 'Pontuação Exame Mental ', 'CDR': 'Grau de Demência'},
                color_discrete_sequence=custom_colors
            )
            fig_strip.update_traces(jitter=0.3)
            st.plotly_chart(fig_strip, use_container_width=True)
        
        with col_graph2:
            # Gráfico de Boxplot
            mmse_doentes = data.loc[(data['Age'] > 60) & (data['CDR'] > 0), ['MMSE']].copy()
            mmse_nao_doentes = data.loc[(data['Age'] > 60) & (data['CDR'] == 0), ['MMSE']].copy()
            
            mmse_doentes['Grupo'] = 'CDR > 0'
            mmse_nao_doentes['Grupo'] = 'CDR = 0'
            dados_comparacao = pd.concat([mmse_doentes, mmse_nao_doentes])
            
            fig_box = px.box(
                dados_comparacao,
                x='Grupo',
                y='MMSE',
                color='Grupo',
                color_discrete_map={'CDR > 0': '#ff7f00', 'CDR = 0': '#4daf4a'},
                title='Distribuição de Pontuação Exame Mental  Entre Doentes e Não Doentes',
                labels={'MMSE': 'Pontuação Exame Mental ', 'Grupo': 'Divisão por Doentes e não Doentes'},       
                )
            fig_box.update_layout(showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)
        
         
        st.subheader("Análise Estatística")
        
        # Configuração do teste
        with st.container():
            col_slider, _ = st.columns([0.3, 0.7])
            with col_slider:
                alpha = st.slider("Nível de significância (α)", 
                                min_value=0.01, max_value=0.10, 
                                value=0.05, step=0.01,
                                help="Limiar para decisão estatística",
                                key="alpha_mmse")
        
        # Layout em colunas para os testes
        col_test1, col_test2 = st.columns(2)
        
        with col_test1:
            # Teste de Normalidade
            with st.expander("Teste de Normalidade (Shapiro-Wilk)", expanded=True):
                stat_d, p_d = shapiro(mmse_doentes['MMSE'])
                stat_nd, p_nd = shapiro(mmse_nao_doentes['MMSE'])
                
                st.markdown("""
                **Hipóteses:**
                - H₀: Os dados seguem uma distribuição normal
                - H₁: Os dados não seguem uma distribuição normal
                """)
                
                st.markdown(f"""
                **Resultados:**
                - **CDR > 0 (Doentes):**
                - Estatística W = `{stat_d:.4f}`, p-valor = `{p_d:.4f}`
                
                - **CDR = 0 (Não Doentes):**
                - Estatística W = `{stat_nd:.4f}`,  p-valor = `{p_nd:.4f}`
                """)
                
                if p_d < 0.05 or p_nd < 0.05:
                    st.warning("""
                    **Conclusão:**  
                    Pelo menos um grupo não segue distribuição normal (p < 0.05).  
                    Recomendado usar teste não-paramétrico.
                    """)
                else:
                    st.success("""
                    **Conclusão:**  
                    Ambos grupos seguem distribuição normal (p ≥ 0.05).  
                    Pode-se usar teste paramétrico.
                    """)
        
        with col_test2:
            # Teste de Mann-Whitney
            with st.expander("Teste de Mann-Whitney U", expanded=True):
                u_stat, p_valor = mannwhitneyu(
                    mmse_doentes['MMSE'],
                    mmse_nao_doentes['MMSE'],
                    alternative='less'
                )
                
                st.markdown("""
                **Hipóteses:**
                - H₀: Não há diferença entre os grupos
                - H₁: CDR > 0 tem MMSE menor (teste unilateral)
                """)
                
                st.markdown(f"""
                **Resultados:**
                - Estatística U = `{u_stat:.2f}`
                - p-valor = `{p_valor:.6f}`
                """)
                
                if p_valor < alpha:
                    st.error(f"""
                    **Conclusão Final:**  
                    Rejeitamos H₀ (p < {alpha})  
                    Há evidências de que pacientes com demência têm MMSE significativamente menor.
                    """)
                else:
                    st.success(f"""
                    **Conclusão Final:**  
                    Não rejeitamos H₀ (p ≥ {alpha})  
                    Não há evidências suficientes para afirmar diferença significativa.
                    """)
        # =============================================
        # Seção 4: Informações Adicionais
        # =============================================
        with st.expander("📌 Sobre a Análise", expanded=False):
            st.markdown("""
            **Metodologia:**
            - População: Pacientes acima de 60 anos
            - Variável resposta: Pontuação Exame Mental (0-30)
            - Grupos comparados: CDR = 0 vs CDR > 0
            - Testes utilizados:
            -- Shapiro-Wilk (normalidade)
            -- Mann-Whitney U (diferença entre grupos)
            
            **Interpretação Clínica:**
            - MMSE < 24 sugere comprometimento cognitivo
            - CDR > 0 indica algum grau de demência
            """)


with tab_pred:
    # Carregar o modelo salvo
    model = joblib.load('decision_tree_model.pkl')

    # Título do app
    st.title("Classificação de Demência (CDR)")

    st.markdown(
        """
        <style>
        .stFrame {
            border: 2px solid #f63366;
            border-radius: 5px;
            padding: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    with st.container():

            st.markdown('<div class="stFrame">', unsafe_allow_html=True)

            st.subheader("Preencha os dados do paciente:")

            # Layout em duas colunas
            col1, col2 = st.columns(2)

            with col1:
                gender = st.selectbox("Sexo (M/F)", ['M', 'F'])
                educ = st.selectbox("Nível educacional", [1, 2, 3, 4, 5])
                age = st.number_input("Idade", min_value=0, max_value=120, value=75)
                etiv = st.number_input("Volume Total Intracraniano Estimado (1100 - 2000)", value=1500.0)

            with col2:
                ses = st.selectbox("Status socioeconômico", [1, 2, 3, 4, 5])
                mmse = st.number_input("Mini-Exame do Estado Mental (0-30) ", min_value=0, max_value=30, value=28)
                nwbv = st.number_input("Volume Normalizado de Matéria Branca (0,63 - 0,85)", value=0.75)

            # Montar DataFrame com os dados inseridos
            input_df = pd.DataFrame({
                'M/F': [gender],
                'Educ': [educ],
                'SES': [ses],
                'Age': [age],
                'MMSE': [mmse],
                'eTIV': [etiv],
                'nWBV': [nwbv]
            })

            # Botão para acionar a previsão
            if st.button("Classificar"):
                prediction = model.predict(input_df)
                proba = model.predict_proba(input_df)
                st.success(f"Resultado previsto (CDR): {prediction[0]}")

            st.markdown('</div>', unsafe_allow_html=True)

with tab_pca:
    def preprocess_data(data):
        dados_pca = data[['M/F', 'Age', 'MMSE', 'ASF', 'nWBV', 'CDR']].copy()
        dados_pca['M/F'] = dados_pca['M/F'].map({'M': 0, 'F': 1})
        return dados_pca
    
    # Aplicar PCA
    def apply_pca(dados_pca):
        dados_normalizados = (dados_pca - dados_pca.mean()) / dados_pca.std()
        pca_3d = PCA(n_components=3)
        pca_resultado = pca_3d.fit_transform(dados_normalizados)
        return pd.DataFrame(data=pca_resultado, columns=['PC1', 'PC2', 'PC3'])

    # Aplicar KMeans
    def apply_kmeans(pca_df, n_clusters=4):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(pca_df)
        pca_df['Cluster'] = labels
        return pca_df

    # Função para criar gráficos
    def plot_3d_scatter(pca_df):
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111, projection='3d')

        # Ordena os grupos para manter a consistência de cores
        grupos = sorted(pca_df['Cluster'].unique())

        cmap = plt.get_cmap("viridis")
        cores = {grupo: cmap(i / len(grupos)) for i, grupo in enumerate(grupos)}

        for grupo in grupos:
            subset = pca_df[pca_df['Cluster'] == grupo]
            ax.scatter(subset['PC1'], subset['PC2'], subset['PC3'], 
                    s=50, alpha=0.6, color=cores[grupo], label=f"Grupo {grupo}")

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title('PCA - Visualização 3D')
        ax.legend()
        return fig


    def plot_single_boxplot(dados_pca, coluna, cmap_name="viridis"):
        fig = plt.figure(figsize=(6, 4))

        grupos = sorted(dados_pca['Cluster'].unique())  # Garante mesma ordem de cor
        cmap = plt.get_cmap(cmap_name)
        cores = {grupo: cmap(i / len(grupos)) for i, grupo in enumerate(grupos)}

        sns.boxplot(x='Cluster', y=coluna, data=dados_pca,
                    palette=[cores[grupo] for grupo in grupos])

        plt.title(f'Distribuição de {coluna} por Cluster')
        plt.tight_layout()
        return fig



    st.title("Dashboard PCA e KMeans")
    st.subheader("Análise de Componentes Principais (PCA) e KMeans")

    st.markdown("""
    **Metodologia:**
    
    Para esta análise, utilizamos as seguintes variáveis do dataset: 
    - Sexo, convertido para valores numéricos: 0 para Masculino, 1 para Feminino
    - Idade
    - Mini Exame do Estado Mental (MMSE)
    - Fator de Escala Atlas (ASF)
    - Volume Cerebral Total Normalizado (nWBV)
    - Escala Clínica de Demência
    
    Foi relizado um pré-processamento dos dados, onde as variáveis foram normalizadas e posteriormente aplicamos a técnica de PCA 
    transformando-as em 3 componentes principais (PC1, PC2 e PC3) que capturam um 80%  da variação nos dados.
    
    Em seguida, aplicamos o algoritmo K-Means com 4 clusters para agrupar os casos com características similares 
    no espaço reduzido pelo PCA.
    
    """)



    data_pca = load_data()
    dados_pca = preprocess_data(data_pca)
    pca_df = apply_pca(dados_pca)
    cluster_pca_df = apply_kmeans(pca_df)
    dados_pca['Cluster'] = cluster_pca_df['Cluster'].values
    
    colunas_boxplot = ['Age', 'MMSE', 'ASF', 'nWBV', 'CDR']


    # Criando duas colunas: a primeira ocupa 40% da largura e a segunda 60%
    col_esquerda, col_direita = st.columns([1, 2])

    # Coluna da esquerda - Apenas o gráfico 3D
    with col_esquerda:
        st.pyplot(plot_3d_scatter(cluster_pca_df))  # Gráfico isolado

    # Coluna da direita - Boxplots em duas linhas
    with col_direita:
        # Primeira linha - 3 boxplots
        col_topo1, col_topo2 = st.columns(2)
        with col_topo1:
            st.pyplot(plot_single_boxplot(dados_pca, colunas_boxplot[0]))
        with col_topo2:
            st.pyplot(plot_single_boxplot(dados_pca, colunas_boxplot[1]))
            

        # Segunda linha - 2 boxplots
        col_base1, col_base2 = st.columns(2)
        with col_base1:
            st.pyplot(plot_single_boxplot(dados_pca, colunas_boxplot[3]))
        with col_base2:
            st.pyplot(plot_single_boxplot(dados_pca, colunas_boxplot[2]))

        col_esq, col, col_dir = st.columns([1, 2, 1])
        with col:
            st.pyplot(plot_single_boxplot(dados_pca, colunas_boxplot[4]))


with tab_simulacao:

    # Função para carregar e preprocessar os dados de simulação
    @st.cache_data
    def load_data_simulacao():
        data = pd.read_csv('arquivos/oasis_cross-sectional.csv')
        data = data.dropna(subset=['MMSE', 'CDR']).drop('Delay', axis=1)
        return data

    # Função para preparar as tabelas de distribuição de CDR por faixa etária
    def preprocess_cdr_tables(data_simulacao):
        data_simulacao_idades_cdr = data_simulacao[['Age', 'CDR']].dropna().reset_index(drop=True)

        # Filtrar apenas idades 65+
        data_65_plus = data_simulacao_idades_cdr[data_simulacao_idades_cdr['Age'] >= 65].copy()

        # Criar faixas etárias
        bins_65_plus = [65, 70, 75, 80, 85, 90, float('inf')]
        labels_65_plus = ['65-69', '70-74', '75-79', '80-84', '85-89', '90+']
        data_65_plus['faixa_etaria'] = pd.cut(data_65_plus['Age'], bins=bins_65_plus, labels=labels_65_plus, right=False)

        # Criar tabela agrupada por faixa etária e CDR
        cdr_faixa_table = data_65_plus.groupby(['faixa_etaria', 'CDR'], observed=True).size().reset_index(name='Count')
        cdr_faixa_table = cdr_faixa_table.pivot(index='faixa_etaria', columns='CDR', values='Count').fillna(0)

        # Filtrar idades >= 60 e CDR > 0
        data_cdr_pos = data_simulacao_idades_cdr[(data_simulacao_idades_cdr['Age'] >= 60) & (data_simulacao_idades_cdr['CDR'] > 0)].copy()

        # Criar faixas etárias ajustadas
        bins_cdr_pos = [60, 70, 80, 90, float('inf')]
        labels_cdr_pos = ['60-69', '70-79', '80-89', '90+']
        data_cdr_pos['faixa_etaria'] = pd.cut(data_cdr_pos['Age'], bins=bins_cdr_pos, labels=labels_cdr_pos, right=False)

        # Criar tabela de porcentagem por faixa etária e CDR
        cdr_faixa_count = data_cdr_pos.groupby(['faixa_etaria', 'CDR'], observed=True).size().reset_index(name='Count')
        total_por_faixa = cdr_faixa_count.groupby('faixa_etaria', observed=True)['Count'].transform('sum')
        cdr_faixa_count['Percent'] = (cdr_faixa_count['Count'] / total_por_faixa) * 100
        cdr_faixa_percent_table = cdr_faixa_count.pivot(index='faixa_etaria', columns='CDR', values='Percent').fillna(0)

        return cdr_faixa_table, cdr_faixa_percent_table

    # Função para calcular a projeção de Alzheimer
    def calcular_projecao_alzheimer():
        populacao_df = pd.read_csv("arquivos/populacao_idosos_2024_2040.csv")
        alzheimer_df = pd.read_csv("arquivos/alzheimer_por_faixa_etaria.csv")

        populacao_long = populacao_df.melt(id_vars="faixa_etaria", var_name="Ano", value_name="Populacao")
        populacao_long = populacao_long.merge(alzheimer_df, on="faixa_etaria")
        populacao_long["Alzheimer_Projecao"] = populacao_long["Populacao"] * (populacao_long["Alzheimer (%)"] / 100)
        populacao_long["Ano"] = populacao_long["Ano"].astype(int)

        return populacao_long

    # Função para visualizar projeção de Alzheimer


    def plot_alzheimer_projection(populacao_long):
        plt.figure(figsize=(12, 6))

        # Especificando a paleta de cores apenas para esse gráfico
        palette = sns.color_palette("dark")  # Troque "magma" por outra paleta se quiser

        # Criando o gráfico sem alterar as configurações globais
        sns.lineplot(data=populacao_long, x="Ano", y="Alzheimer_Projecao", hue="faixa_etaria", 
                    marker="o", palette=palette)

        plt.title("Projeção de Pessoas com Alzheimer por Faixa Etária (2024–2040)")
        plt.xlabel("Ano")
        plt.ylabel("Número Estimado de Pessoas com Alzheimer")
        plt.legend(title="Faixa Etária", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        st.pyplot(plt)

    # Função para calcular projeção por CDR
    def calcular_projecao_cdr(populacao_long, cdr_faixa_percent_table):
        alz_idade_ano = populacao_long[['Ano', 'faixa_etaria', 'Alzheimer_Projecao']].copy()

        alz_idade_ano["faixa_etaria_ajustada"] = alz_idade_ano["faixa_etaria"].replace({
            "65-69": "60-69", "70-74": "70-79", "75-79": "70-79",
            "80-84": "80-89", "85-89": "80-89", "90+": "90+"
        })

        df_final = alz_idade_ano.merge(cdr_faixa_percent_table, left_on="faixa_etaria_ajustada", right_on="faixa_etaria", how="left")

        for cdr in [0.5, 1.0, 2.0]:
            df_final[f"CDR {cdr} Projecao"] = (df_final["Alzheimer_Projecao"] * (df_final[cdr] / 100)).round(0).astype(int)

        df_agrupado = df_final.groupby("Ano")[["CDR 0.5 Projecao", "CDR 1.0 Projecao", "CDR 2.0 Projecao"]].sum()
        
        return df_agrupado

    # Função para visualizar projeção por CDR
    def plot_cdr_projection(df_agrupado):
        # Definindo as cores personalizadas para cada categoria
        cores = {
            "CDR 0.5 Projecao": '#ff9999',  # Vermelho leve
            "CDR 1.0 Projecao": '#e41a1c',  # Vermelho intermediário
            "CDR 2.0 Projecao": '#990000'   # Vermelho intenso
        }

        # Criando a figura
        plt.figure(figsize=(10, 5))

        # Plotando cada linha com a cor correspondente
        for col in cores.keys():
            plt.plot(df_agrupado.index, df_agrupado[col], label=col, color=cores[col], linewidth=2)

        # Adicionando os rótulos e título
        plt.xlabel("Ano")
        plt.ylabel("Quantidade de Casos")
        plt.title("Evolução do número de casos de Alzheimer por gravidade")

        # Exibindo a legenda e grid
        plt.legend()
        plt.grid()

        # Renderizando o gráfico no Streamlit
        st.pyplot(plt)

    # Configuração do Streamlit
    
    data_simulacao = load_data_simulacao()
    cdr_faixa_table, cdr_faixa_percent_table = preprocess_cdr_tables(data_simulacao)
    populacao_long = calcular_projecao_alzheimer()
    df_agrupado = calcular_projecao_cdr(populacao_long, cdr_faixa_percent_table)

    # tabelas e grafico principal
    st.subheader("Projeção de Pessoas com Alzheimer no Brasil (2024-2040)")

    alzheimer_por_idade_brasil = pd.read_csv("arquivos/alzheimer_por_faixa_etaria.csv")

    # Cria duas colunas com proporção 1:3
    col1, col2 = st.columns([1, 3])

    # Na primeira coluna (mais estreita), mostra a tabela
    with col1:
        st.subheader("Porcentagem de Pessoas com Alzheimer por Faixa Etária")
        st.dataframe(
            alzheimer_por_idade_brasil,
            height=300,  # Altura fixa para melhor visualização
            hide_index=True,  # Oculta o índice se não for relevante
            use_container_width=True  # Usa toda a largura da coluna
        )

    # Na segunda coluna (mais larga), mostra o gráfico
    with col2:
        plot_alzheimer_projection(populacao_long)

    # Criar duas colunas abaixo do gráfico principal
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Evolução do Número de Casos por Gravidade")
        plot_cdr_projection(df_agrupado)

    with col2:
        st.subheader("Tabela de Projeção por Demência Clinica") 
        st.write(df_agrupado)


with tab_conclusoes:
    st.subheader("Conclusões")

    with st.container(border=True): 
        st.markdown("""
        <style>
            .custom-text {
                font-size: 20px;
                font-weight: bold;
            }
        </style>
        <div class="custom-text">
        - Resultados baixos no Mini Exame de Estado Mental são um sinal de alerta para possíveis casos de demência.<br>
        - Pode-se considerar realizar o MMSE a partir dos 60 anos.<br>
        - Exames de imagem são recomendados para fornecer uma conclusão após os resultados do Exame de Estado Mental.<br>
        - São necessarios políticas públicas para aumentar a conscientização sobre a demência e o Alzheimer, especialmente entre os idosos, para promover um diagnóstico precoce e intervenções adequadas.
        </div>
        """, unsafe_allow_html=True)