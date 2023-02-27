# generate requirements.txt
requirements:
	pip freeze > requirements.txt

test:
	pytest -s -v

commit: requirements
	git add .
	git commit -a 

commit-and-push: commit
	git push