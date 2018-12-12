docker build -t "attngan" -f dockerfile.cpu .
docker run -it --env-file test.env -p 5678:8080 attngan
