# Input Parsing Integration Module

## Overview
This folder contains the **System Architect** component of the **Multi-Agent Emergency Response System**. The module is responsible for connecting natural language emergency input with the central dispatch system.

It acts as the integration layer between:
- user emergency text input
- emergency type and location extraction
- the dispatch module
- future routing integration

## Purpose
The goal of this module is to transform raw emergency descriptions into a structured format that can be passed into the dispatch system. This allows the system to process human-readable input and generate a coordinated emergency response.

## Features
- Parses natural language emergency input
- Extracts emergency type from text
- Extracts location from text
- Generates structured input for downstream modules
- Integrates with the central `dispatch()` function
- Supports end-to-end emergency response testing

## Example Input
```text
Fire near Docklands
