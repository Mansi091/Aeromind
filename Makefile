.PHONY: up down build logs test

# Start the Docker containers in the background
up:
	docker compose up -d

# Build and start the containers (useful if you changed requirements.txt or Dockerfile)
build:
	docker compose up --build -d

# Stop and remove the containers
down:
	docker compose down

# View live logs from the containers
logs:
	docker compose logs -f

# Run the pytest suite locally
test:
	pytest
