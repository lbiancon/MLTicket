.\.venv\Scripts\activate
python src\generate_dataset.py
python src\train_eval.py
python -m streamlit run src\app.py