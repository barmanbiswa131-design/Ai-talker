# BISWA AI

Private-data-minimizing AI generation API project.

## Goal
Photo generation/editing, image-to-video, dance/action motion, character consistency, and later audio/song generation.

This repository contains the API, inference/training scaffolding, evaluation tools, and configuration. It does not store user media permanently.

## Important
The repository is a starting engineering/training framework. High-quality video generation requires a compatible open-source base model, GPU compute, licensed training data, and substantial training/inference resources.

## Structure
- api/ — FastAPI service
- training/ — dataset/training configuration and scripts
- inference/ — model adapter
- privacy/ — temporary-file lifecycle
- tests/ — API and privacy tests
