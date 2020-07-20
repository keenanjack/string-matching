dev:
	FLASK_ENV=development python app.py

docker_run:
	docker run -p 5000:80 stringmatching