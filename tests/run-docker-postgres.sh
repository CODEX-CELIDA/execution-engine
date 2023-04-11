#!/bin/bash
docker run --name postgres-pytest -e POSTGRES_PASSWORD=mysecretpassword -p 5433:5432 postgres
