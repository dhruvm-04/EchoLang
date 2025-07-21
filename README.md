# EchoLang

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Transformers](https://img.shields.io/badge/Transformers-4.30%2B-orange)

Real-time Speech-to-Text (STT) and translation system for code-switched Hindi-Tamil-English audio. Optimized for low-resource settings, it enables contextual transcription and intent-aware translation for use cases like voice-based search and medical transcription.

## Problem Statement
Real time speech to text (STT) translation and transcription tool, focused on a mix of Hindi/Tamil with English. (Transcription and translation of mixed Hindi-English (Hinglish) and Tamil-English speech into English text.)
Considering the use case - Accessing a Yellow Page Directory for blue collar workers living in Tier 2/3 cities of India, that can speak a mix of **English/Hindi/Tamil**

## Overview

EchoLang is an intelligent multilingual service request analysis system designed to process and categorize service requests in **Hindi**, **Tamil**, and **English**. The system combines advanced speech recognition, neural machine translation, and intent classification to provide a comprehensive solution for service request management.

## Key Features

- Speech-to-Text: Advanced ASR using Whisper with n-gram language model fusion
- Multilingual Support: Native support for Hindi, Tamil, and English
- Neural Translation: NLLB-200 powered translation for cross-language understanding
- Intent Classification: Automated service category detection with confidence scoring
- Interactive Chatbot: Conversational interface for guided service request collection
- Real-time Analysis: Instant processing with detailed confidence metrics
- Service Categories: Emergency Services, Healthcare, Home Maintenance, Transportation, Cleaning, and General Services

## Prerequisites
- pip install torch>=2.0.0 transformers>=4.30.0 openai-whisper>=20231117
- pip install sentence-transformers>=2.2.0 langdetect>=1.0.9 jiwer>=3.0.0
- pip install soundfile>=0.12.0 librosa>=0.10.0 accelerate>=0.20.0

## Project Structure
  EchoLang/  
  ├── docs/ # Documentation  
  ├── src/ # Source code  
  │ ├── models/ # Model management  
  │ ├── processing/ # Audio/text processing  
  │ ├── translation/ # Translation components  
  │ ├── classification/ # Intent classification  
  │ └── ui/ # User interface  
  ├── tests/ # Unit tests  
  ├── examples/ # Usage examples  
  └── notebooks/ # Jupyter notebooks  
