# Kyber

Futuristic, professional static download page for Kyber. This project currently hosts a polished landing page where official installers will be published when available.

Status
- Pre-release: official binaries for Windows, Linux, and macOS are not yet published. Download buttons are intentionally disabled.

Files
- `index.html` — modern, responsive landing page (futuristic design). Use this as the public download page.
- `.gitignore` — common ignores

View locally
1. Open `index.html` in a browser.

Publish (recommended)
- Use GitHub Pages to host this site for free:
  1. Create a GitHub repo (example: `yourusername/kyber`).
  2. Push this folder to the repo and enable GitHub Pages on the main branch (root).
  3. The site will be available at `https://yourusername.github.io/kyber/`.

Want help publishing?
- Reply with one of the following and I'll proceed:
  - "Create + publish to GitHub Pages" — provide the desired repo name (e.g., `kyber`) and a GitHub token with repo permissions OR add the remote and I will push.
  - "Create repo only" — provide repo name; I'll create files and you push.
  - "Do not publish" — I'll leave files local.

Notes
- Download buttons are disabled until official artifacts are published. If you need builds now, follow the build-from-source steps in the page.

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
