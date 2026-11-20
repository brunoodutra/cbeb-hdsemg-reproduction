# CBEB HD-sEMG Reproduction

Reprodutibilidade dos experimentos do artigo CBEB 2026 sobre HMIs baseadas em
sEMG de alta densidade (HD-sEMG) — classificação de gestos com canais ruidosos/faulty.

## Estrutura

- `DataBase_Treatment/` — tratamento das bases de dados (inclui pipeline da base 65 HD EMG)
- `Dataloader/` — generators e pipelines de dados (Keras)
- `Experiments/` — experimentos de machine learning e deep learning e análise do paper
- `Keras_deep_models/` — arquiteturas dos modelos (Conv2D ResNet)
- `Processing/` — pré-processamento e pipeline de processamento do sinal
- `Utils/` — métricas e funções utilitárias
