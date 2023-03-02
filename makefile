# Determine platform-specific requirements file name
UNAME := $(shell uname)
ifeq ($(UNAME),Darwin)
    REQUIREMENTS_FILE := requirements-mac.txt
else ifeq ($(UNAME),Linux)
    REQUIREMENTS_FILE := requirements-linux.txt
else ifeq ($(UNAME),Windows)
    REQUIREMENTS_FILE := requirements-windows.txt
else
    REQUIREMENTS_FILE := requirements.txt
endif

# Freeze python environment into platform-specific requirements file
generate-requirements:
	pip freeze > $(REQUIREMENTS_FILE)

# Install python environment from platform-specific requirements file
install-requirements:
	pip install -r $(REQUIREMENTS_FILE)

# Run tests using pytest
test:
	pytest -s -v

# Commit changes and update requirements.txt
commit: requirements
	git add .
	git commit -a 

# Commit changes, update requirements.txt, and push to remote repository
commit-and-push: commit
	git push
