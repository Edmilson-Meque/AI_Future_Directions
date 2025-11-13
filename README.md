# AI Future Directions - Edge AI Demo

## Projeto: Edge AI para Classificação de Recicláveis

Este projeto demonstra como criar e converter um modelo de **Edge AI** usando TensorFlow Lite para dispositivos de baixa capacidade, como Raspberry Pi ou smartphones.

### Objetivos
- Treinar um modelo de classificação de imagens leve.
- Converter o modelo para TensorFlow Lite (`.tflite`).
- Testar em uma imagem de exemplo.
- Demonstrar como Edge AI reduz latência e aumenta privacidade.

### Estrutura do Repositório
- `notebooks/edge_ai_demo.ipynb` → notebook para treinar e converter o modelo.
- `src/edge_ai_model.py` → funções de pré-processamento e inferência.
- `data/sample_dataset/` → dataset de imagens de recicláveis.
- `diagrams/edge_ai_flowchart.png` → diagrama do fluxo de Edge AI.
- `requirements.txt` → dependências do projeto.

### Como Executar
1. Instale dependências:
```bash
pip install -r requirements.txt
