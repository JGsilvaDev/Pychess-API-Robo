# ♟️ Pychess-API

Pychess-API é uma API desenvolvida com FastAPI para fornecer funcionalidades relacionadas ao xadrez, como validação de lances, análise de jogadas e gerenciamento de partidas.  
Ela já está integrada com o mecanismo de xadrez **Stockfish**, permitindo análises automatizadas de jogadas.

---

## 🚀 Configuração e Execução

Para rodar o projeto, siga os passos abaixo:

### 1. Criar um ambiente virtual  
```sh
python -m venv venv
```

### 2. Ativar o ambiente virtual  
- **Windows**:  
  ```sh
  venv\Scripts\activate
  ```
- **Linux/Mac**:  
  ```sh
  source venv/bin/activate
  ```

### 3. Instalar as dependências  
```sh
pip install fastapi uvicorn
```

### 4. Iniciar o servidor  
```sh
uvicorn main:app --reload
```

## Acessando a Documentação da API  

Após iniciar o servidor, acesse a interface interativa do Swagger para visualizar e testar as APIs:  
🔗 **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

---

📌 **Tecnologias utilizadas:**  
✅ Python
✅ FastAPI
✅ Uvicorn
✅ Stockfish

🤝 Contribuindo
Contribuições são bem-vindas!
Sinta-se à vontade para abrir issues, propor melhorias ou enviar pull requests.

