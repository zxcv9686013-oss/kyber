# Kyber

This repository contains a simple static download page for Kyber and basic documentation.

Note: Pre-built installers for Windows, Linux, and macOS are not available yet. The download buttons on the web page are intentionally disabled.

Files added:
- `index.html` — static webpage with disabled download buttons

How to view locally:
1. Open `index.html` in a browser.

How to publish on GitHub Pages:
1. Create a GitHub repo (example: `yourusername/kyber`).
2. Push this repository to GitHub:
   - `git remote add origin https://github.com/yourusername/kyber.git`
   - `git branch -M main`
   - `git push -u origin main`
3. Enable GitHub Pages from repository settings (use main branch / root).

If you want, provide a GitHub repo URL or permission and I can push these files to GitHub for you.
## Dependencies

Kyber requires the following tools for building and/or using its supported backends:

- CMake
- Ninja
- CUDA
- XLA
- DirectML
- Clang/LLVM
- NASM

These dependencies are automatically installed and configured by the Kyber installer.

The third-party dependency files are not committed to this repository because we respect their respective licenses and redistribution limitations. Kyber uses and configures these dependencies in accordance with their applicable licensing terms.
