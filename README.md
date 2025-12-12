# Disaster Message Classification and Response System
This repository contains our NLP/ML pipeline for classifying real-world disaster response messages. The goal is to support emergency-response workflows by automatically identifying critical needs such as food, water, shelter, and aid-related requests from unstructured text.

The system includes:
- A preprocessing and feature-extraction pipeline
- Multiple classical ML models for multi-label classification
- Evaluation metrics and visualizations

## Dataset
This dataset contains tens of thousands of real disaster-related messages labeled across multiple humanitarian categories including food, water, shelter, medical aid, and more.

Dataset being used: https://huggingface.co/datasets/community-datasets/disaster_response_messages

## Backend API for Message Classification
This repo also includes a production-ready backend of our NLP project on learning and Classifying Disaster Response Messages. We created an API using Flask to be able to deploy and call our machine learning models.
