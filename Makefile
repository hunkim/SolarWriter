# Define the two virtual environments
VENV = .venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip3
STREAMLIT = $(VENV)/bin/streamlit

ENV_FILE := .env

# Load environment variables
include $(ENV_FILE)
export

# Browser-enabled venv with playwright
$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt
	$(PIP) install playwright
	$(PYTHON) -m playwright install

app: $(VENV)/bin/activate
	$(STREAMLIT) run app.py

dp_test: $(VENV)/bin/activate
	$(PYTHON) dp.py

clean:
	rm -rf __pycache__
	rm -rf $(VENV)
