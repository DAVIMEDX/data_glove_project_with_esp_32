# 🧤 Reconhecimento de Língua de Sinais com Luva Instrumentada e Deep Learning

## 📖 Sobre o Projeto
Este repositório contém o código-fonte e os experimentos de Machine Learning para a classificação de 40 gestos de Língua de Sinais utilizando uma luva instrumentada. O sistema processa séries temporais contínuas capturadas por sensores de flexão (dedos) e sensores inerciais (IMU - acelerômetro e giroscópio) acoplados à mão do usuário.

O foco principal do projeto é avaliar diferentes arquiteturas de redes neurais artificiais para identificar a melhor relação entre desempenho preditivo e capacidade de modelagem temporal (visando aplicações futuras em sistemas embarcados/TinyML).

---

## ⚙️ Pré-Processamento e Metodologia

Trabalhar com biometria e sensores vestíveis (*wearables*) exige um tratamento cuidadoso dos dados. Para este projeto, implementamos:

* **Validação LOSO (Leave-One-Subject-Out):** Para garantir que o modelo não está apenas "decorando" os participantes do treino, a validação isola os sujeitos, testando a capacidade real de generalização da rede para pessoas inéditas.
* **Normalização Híbrida Específica por Sujeito:** Uma etapa crucial de calibração automática de hardware.
  * **Sensores de Flexão:** Utilizam `MinMaxScaler` para respeitar os limites anatômicos da mão (0 a 1).
  * **IMU (Acelerômetro/Giroscópio):** Utilizam `StandardScaler` (Z-score) para lidar com a natureza dinâmica e estabilizar picos de aceleração abruptos.
  * *Nota:* Essa normalização é aplicada individualmente por usuário para mitigar deslocamentos (*hardware offsets*) causados pelo tamanho da mão e ajuste da luva.

---

## 📊 Arquiteturas Avaliadas e Resultados

O projeto comparou quatro arquiteturas base: um modelo denso atuando como *baseline* estático e três modelos sequenciais nativos. A avaliação considerou a métrica de F1-Score (Macro) para 40 classes sob a rigorosa validação LOSO.

| Arquitetura | Papel | Parâmetros | Acurácia | F1-Score (Macro) |
| :--- | :--- | :--- | :--- | :--- |
| **CNN 1-D** | Sequencial | 39.528 | 0,816 | 0,8061 |
| **MLP** | *Baseline* | 184.808 | 0,810 | 0,7757 |
| **GRU** | Sequencial (Adotado) | 28.904 | 0,798 | 0,7658 |
| **LSTM** | Sequencial | 36.584 | 0,807 | 0,7449 |

### 🏆 Escolha do Modelo: GRU
Embora a rede convolucional (CNN 1-D) tenha alcançado o maior pico de pontuação isolada, **a arquitetura GRU foi selecionada como a principal do projeto**. O modelo GRU entrega um F1-Score robusto de aproximadamente 76,5% utilizando apenas 28 mil parâmetros (o modelo mais leve do experimento). Além disso, sua capacidade de manter dependências temporais contínuas o torna a escolha fisiologicamente mais coerente para o rastreamento do movimento humano do início ao fim do gesto.

---

## 🚀 Como Executar Localmente (Treinamento e Inferência)

### 1. Pré-requisitos
Certifique-se de ter o **Python 3.11** instalado em sua máquina. O uso de versões muito recentes (como 3.13) ou ambientes gerenciados pelo `uv` podem bloquear a instalação global de pacotes devido à PEP 668.

### 2. Configurando o Ambiente e Instalando Dependências
Clone o repositório e crie uma "bolha" isolada para o projeto usando o terminal:

```bash
# Clone o repositório
git clone https://github.com/DAVIMEDX/data_glove_project_with_esp_32.git
cd data_glove_project_with_esp_32

# Crie o ambiente virtual (Windows)
py -3.11 -m venv .venv

# Ative o ambiente virtual
.\.venv\Scripts\activate

# Instale as dependências necessárias para o projeto
pip install -r requirements.txt
