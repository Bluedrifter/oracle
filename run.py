from app import create_app

app = create_app()

if __name__ == "__main__":
    # 127.0.0.1 on purpose. the oracle is wise, not reckless.
    app.run(debug=True, host="127.0.0.1", port=5000)
