# Guia de Migrations com Alembic

## Requisitos
Antes de começar, certifique-se de que você tem o Alembic e o SQLAlchemy instalados no seu ambiente:

```bash
pip install alembic sqlalchemy
```

## 1. Inicializar o Alembic no Projeto
No diretório do seu projeto, execute:

```bash
alembic init alembic
```

Isso criará uma pasta `alembic/` com a estrutura necessária para gerenciar as migrations.

## 2. Configurar o Alembic
Edite o arquivo `alembic.ini` e configure a string de conexão com o banco de dados:

```ini
sqlalchemy.url = sqlite:///database.db  # Altere conforme necessário
```

No arquivo `alembic/env.py`, substitua:

```python
from models import Base  # Importe a Base do seu projeto

target_metadata = Base.metadata
```

## 3. Criar a Primeira Migration
Se você já tem modelos SQLAlchemy definidos, gere a primeira migration automaticamente:

```bash
alembic revision --autogenerate -m "initial migration"
```

Se quiser criar manualmente:

```bash
alembic revision -m "create tables"
```

Isso criará um arquivo em `alembic/versions/` com as instruções de criação de tabelas.

## 4. Aplicar as Migrations
Para aplicar as migrations e criar as tabelas no banco de dados:

```bash
alembic upgrade head
```

## 5. Criar Novas Migrations
Sempre que modificar os modelos, gere uma nova migration:

```bash
alembic revision --autogenerate -m "update models"
```

Depois, aplique:

```bash
alembic upgrade head
```

## 6. Resetar o Banco de Dados (Opcional)
Para limpar e recriar as tabelas:

```bash
alembic downgrade base
alembic upgrade head
```

## 7. Verificar o Status das Migrations
Para ver quais migrations já foram aplicadas:

```bash
alembic history
```

Para ver a migration atual:

```bash
alembic current
```
