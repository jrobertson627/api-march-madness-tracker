"""Entry point. `gunicorn app:app` in production, `python app.py` for local development."""

import os

from api.app import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
