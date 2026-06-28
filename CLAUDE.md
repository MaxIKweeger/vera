# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This directory is a personal Claude Code workspace, not a full software project. It currently contains:

- `claude.bat` — Windows launcher that runs `claude.exe` with `--permission-mode bypassPermissions`
- `noirlab2521aj.tif` — A TIFF image file (NOIRLab astronomical image)

## Launcher

`claude.bat` starts Claude Code without permission prompts:

```bat
"C:\Users\hugues\.local\bin\claude.exe" --permission-mode bypassPermissions
```
