# Projeto 2 MS/MT-571

<!--toc:start-->
- [Projeto 2 MS/MT-571](#projeto-2-msmt-571)
  - [Para rodar o projeto](#para-rodar-o-projeto)
    - [Requisitos](#requisitos)
    - [Instruções](#instruções)
<!--toc:end-->

## Para rodar o projeto

### Requisitos

- `python 3.10+`

  - `pip`

  - `venv`

### Instruções

Clone o repositório e faça:

```bash
python3 -m venv .env
source ./.env/bin/activate
pip install -r requirements.txt
python -m ipykernel install --name=.env
jupyter lab <notebook>.ipynb
```

Em seguida selecione o *kernel* `.env` e execute as células do notebook (note que já apresentamos nossa última execução).

Se forem feitas mudanças no notebook que devem ser salvas, faça:

```bash
jupytext <notebook>.ipynb --to py:percent
```

